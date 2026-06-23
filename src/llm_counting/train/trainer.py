import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Any, Dict
import math
from einops import rearrange
from dataclasses import dataclass
import torch.nn.functional as F 
import dataclasses
from torch import optim 
import matplotlib.pyplot as plt 

BOS_ID =  10
VOCAB=11

def sample_batch(batch_size:int, L:int, D=10, device='cpu'):
  n, rem = divmod(L, D)
  assert rem == 0, "L must be divisible by D"

  base = torch.arange(D).repeat_interleave(n)
  perms = torch.rand(batch_size, L).argsort(dim=-1)
  seqs = base[perms]

  bos = torch.full((batch_size, 1), BOS_ID, dtype=torch.long)
  seqs = torch.cat([bos, seqs], dim=1)
  inputs = seqs[:, :-1]
  targets = seqs[:, 1:]
  return inputs.to(device), targets.to(device)

def true_conditional(targets:torch.Tensor, VOCAB:int=11):
  '''
  targets [BATCH, SEQ_LEN]
  '''
  batch_size, seq_len = targets.shape
  n = seq_len  // (VOCAB-1)
  one_hot_targets = F.one_hot(targets, num_classes=VOCAB-1)
  cumsum = torch.cumsum(one_hot_targets, dim=1) - one_hot_targets
  remaining = (n-cumsum).clamp(min=0)
  return remaining / remaining.sum(-1, keepdim=True)

@torch.no_grad()
def entropy_floor(n_batches=200, batch_size=256, seq_len=100, D=10, device="cpu"):
    """
    Bayes-optimal per-token cross-entropy for the balanced task, in nats.
    Self-contained: samples sequences, builds their true r_d/R conditionals, and
    averages the per-position entropy. Depends only on (seq_len, D), not on any model.
    """
    total = 0.0
    for _ in range(n_batches):
        _, targets = sample_batch(batch_size, seq_len, D=D, device=device)
        p = true_conditional(targets, VOCAB=D + 1)        # (B, L, D)
        ent = torch.special.entr(p).sum(dim=-1)           # (B, L) per-position entropy, 0·log0=0
        total += ent.mean().item()                        # mean over all B*L positions
    return total / n_batches

from pathlib import Path

from llm_counting.model.model import Transformer


@dataclass
class TrainingArgs:
    batch_size: int = 64
    num_steps: int = 2000
    seq_length: int = 100
    checkpoint_every_step: int = 10
    device: str = "cpu"
    lr: float = 1e-4
    val_every: int = 10
    log_every: int = 10
    weight_decay: float = 0.1
    save_checkpoints: bool = True      # NEW: turn off for fast test runs
    probe_size: int = 4                # NEW: fixed probe set for distribution plots


@dataclass
class ModelArgs:
    input_dim: int = 64
    out_dim: int = 11
    attn_dim: int = 64
    hidden_dim: int = 64
    num_heads: int = 4
    causal: bool = True
    max_len: int = 512
    num_blocks: int = 2
    VOCAB: int = 11

    def __post_init__(self):
        assert self.attn_dim % self.num_heads == 0, \
            "attn_dim must divide evenly by num_heads"
        assert (self.attn_dim // self.num_heads) % 2 == 0, \
            "per-head dim (attn_dim // num_heads) must be even for RoPE"


class Train:
    def __init__(self, args: TrainingArgs, model_args: ModelArgs):
        self.args = args
        self.model_args = model_args
        self.device = torch.device(args.device)

        # --- cross-arg validation ---
        assert args.batch_size > 0 and args.num_steps > 0 and args.lr > 0
        assert args.seq_length % 10 == 0, \
            "seq_length must be divisible by the number of digits (10)"
        assert args.seq_length + 1 <= model_args.max_len, \
            f"seq_length+1 ({args.seq_length + 1}) exceeds max_len ({model_args.max_len})"

        # --- build the model ---
        cfg = dataclasses.asdict(model_args)
        cfg.pop("VOCAB", None)
        self.model = Transformer(**cfg).to(self.device)

        # --- optimizer ---
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # --- baseline ---
        self.entropy_floor = entropy_floor(
            seq_len=args.seq_length, batch_size=args.batch_size, device=args.device
        )

        # --- bookkeeping ---
        self.step = 0
        self.best_val = float("inf")
        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(exist_ok=True)
        self.history = {
            "train_step": [], "train_loss": [],
            "val_step": [], "val_ce": [], "gap": [], "pos_kl": [],
        }

        # --- fixed probe set for distribution plots ---
        self.probe_inputs, self.probe_targets = sample_batch(
            args.probe_size, args.seq_length, device=self.device
        )
        self.probe_true = true_conditional(self.probe_targets, VOCAB=model_args.out_dim)  # (P, L, D)
        self.probe_history = []   # list of (step, probs) with probs on cpu

    def training_step(self, inputs, targets):
        self.optimizer.zero_grad()
        logits = self.model(inputs)                       # (B, S, V)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def validation(self, n_batches: int = 20):
        self.model.eval()
        D = self.model_args.out_dim - 1
        total_ce = 0.0
        pos_kl = torch.zeros(self.args.seq_length, device=self.device)
        for _ in range(n_batches):
            inputs, targets = sample_batch(
                self.args.batch_size, self.args.seq_length, device=self.device
            )
            logits = self.model(inputs)                   # (B, L, V)
            V = logits.size(-1)
            total_ce += F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1)).item()

            # per-position KL(true ‖ model) on the digit distribution
            masked = logits.clone()
            masked[..., -1] = float("-inf")               # drop BOS so digits sum to 1
            p_model = torch.softmax(masked, dim=-1)[..., :D]          # (B, L, D)
            p_true = true_conditional(targets, VOCAB=self.model_args.out_dim)  # (B, L, D)
            kl = (torch.xlogy(p_true, p_true) - torch.xlogy(p_true, p_model)).sum(-1)  # (B, L)
            pos_kl += kl.mean(0)
        self.model.train()

        val_ce = total_ce / n_batches
        pos_kl = (pos_kl / n_batches).cpu()
        gap = None if self.entropy_floor is None else val_ce - self.entropy_floor
        return {"val_ce": val_ce, "gap_to_floor": gap, "pos_kl": pos_kl}

    @torch.no_grad()
    def snapshot_probe(self):
        was_training = self.model.training
        self.model.eval()
        logits = self.model(self.probe_inputs).clone()    # (P, L, V)
        logits[..., -1] = float("-inf")                   # mask BOS
        probs = torch.softmax(logits, dim=-1)[..., :self.model_args.out_dim - 1]  # (P, L, D)
        self.probe_history.append((self.step, probs.cpu()))
        if was_training:
            self.model.train()

    def save_checkpoint(self, tag):
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "args": dataclasses.asdict(self.args),
                "model_args": dataclasses.asdict(self.model_args),
            },
            self.ckpt_dir / f"{tag}.pt",
        )

    def training_loop(self, overfit_one_batch=False):
        self.model.train()
        fixed = (sample_batch(self.args.batch_size, self.args.seq_length, device=self.device)
                 if overfit_one_batch else None)
        do_save = self.args.save_checkpoints and not overfit_one_batch
        pbar = tqdm(range(self.args.num_steps))
        for step in pbar:
            self.step = step
            inputs, targets = fixed if overfit_one_batch else sample_batch(
                self.args.batch_size, self.args.seq_length, device=self.device)
                # TRY loss function on the conditional
            loss = self.training_step(inputs, targets)

            if step % self.args.log_every == 0:
                self.history["train_step"].append(step)
                self.history["train_loss"].append(loss)
                pbar.set_postfix(loss=f"{loss:.4f}")

            if step % self.args.val_every == 0:
                m = self.validation()
                self.history["val_step"].append(step)
                self.history["val_ce"].append(m["val_ce"])
                self.history["gap"].append(m["gap_to_floor"])
                self.history["pos_kl"].append(m["pos_kl"].numpy())
                self.snapshot_probe()

                msg = f"step {step} | train {loss:.4f} | val {m['val_ce']:.4f}"
                if m["gap_to_floor"] is not None:
                    msg += f" | gap {m['gap_to_floor']:.4f}"
                pbar.write(msg)

                if do_save and m["val_ce"] < self.best_val:
                    self.best_val = m["val_ce"]
                    self.save_checkpoint("best")

            if do_save and step % self.args.checkpoint_every_step == 0:
                self.save_checkpoint(f"step_{step}")

    # ---------- plots ----------
    def plot_loss(self, path="loss.png"):
        h = self.history
        fig, ax = plt.subplots()
        ax.plot(h["train_step"], h["train_loss"], label="train", alpha=0.5)
        ax.plot(h["val_step"], h["val_ce"], label="val", marker=".")
        ax.axhline(self.entropy_floor, ls="--", c="gray",
                   label=f"floor {self.entropy_floor:.3f}")
        ax.set(xlabel="step", ylabel="cross-entropy (nats)")
        ax.legend()
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    def plot_gap(self, path="gap.png"):
        h = self.history
        steps = [s for s, g in zip(h["val_step"], h["gap"]) if g is not None and g > 0]
        gaps = [g for g in h["gap"] if g is not None and g > 0]
        fig, ax = plt.subplots()
        ax.plot(steps, gaps, marker=".")
        ax.set(xlabel="step", ylabel="val_CE − floor  (≈ mean KL)", yscale="log")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    def plot_pos_kl(self, path="pos_kl.png"):
        fig, ax = plt.subplots()
        ax.plot(self.history["pos_kl"][-1])
        ax.set(xlabel="position", ylabel="KL(true ‖ model)", title="latest validation")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

    def plot_depletion(self, seq_idx=0, digit=0, n_snaps=5, path="depletion.png"):
        gold = self.probe_true[seq_idx, :, digit].cpu()
        snaps = self.probe_history
        idx = torch.linspace(0, len(snaps) - 1, n_snaps).long().tolist()
        fig, ax = plt.subplots()
        ax.plot(gold, "k--", lw=2, label="true r_d/R")
        for i in idx:
            step, probs = snaps[i]
            ax.plot(probs[seq_idx, :, digit], alpha=0.7, label=f"step {step}")
        ax.set(xlabel="position", ylabel=f"P(digit {digit})")
        ax.legend()
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

