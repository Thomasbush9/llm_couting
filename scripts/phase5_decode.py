"""Phase 5 - where the (1-eps)^L decoding erosion enters the circuit.

All earlier probing is on teacher-forced VALID sequences. Here we probe the model's
OWN free-running rollouts (greedy + ancestral T=1) to localise the failure:

  (A) Count tracking on own rollouts: apply the teacher-forced count probe to the
      residual of generated sequences and compare to the true running count. If the
      count stays accurate even as sampled sequences drift out of distribution, the
      counting REPRESENTATION is not the failure point.

  (B) The leak at the converter: at each step, the probability mass the model places on
      already-DEPLETED digits (true remaining = 0). Bayes says this must be 0 (logit
      = log 0 = -inf), but a real MLP can only reach a finite floor -> a small eps>0.
      Greedy never picks it (lowest prob); sampling occasionally draws it.

  (C) Survival accounting: from the per-step leak eps(t) on still-valid rollouts, predict
      sequence survival prod(1-eps(t)) and compare to the empirical fraction of valid
      sampled rollouts, at L in {50,100,200,300}. A match pins the length-dependent
      validity loss (Phase 15.4 / Sec.13) to the converter's depletion floor.

Outputs: phase5_decode.png  (+ printed table).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.train.trainer import sample_batch
from scripts.probing_snippets import load_model, cache_resids, counts_from_targets

device = "cuda"; D = 10; VOCAB = 11; BOS = 10
CKDIR = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50")
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def fit_count_probe(model, L, bs=256):
    x, y = sample_batch(bs, L, device=device)
    X = cache_resids(model, x)["mid1"][:, 1:].reshape(-1, 64).numpy()
    c = counts_from_targets(y).cpu()[:, 1:].reshape(-1, D).numpy()
    return Ridge(1.0).fit(X, c)


@torch.no_grad()
def rollout(model, L, mode, bs=512, T=1.0):
    seq = torch.full((bs, 1), BOS, dtype=torch.long, device=device)
    for _ in range(L):
        lg = model(seq)[:, -1, :].clone(); lg[:, BOS] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq[:, 1:]                                       # (bs,L) generated digits


@torch.no_grad()
def analyse(model, probe, L, gen):
    """Given generated digits (bs,L): per-position count-MAE, leak, and survival."""
    n = L // D; bs = gen.shape[0]
    inp = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long, device=device), gen[:, :-1]], 1)
    # residual + logits in one teacher-forced pass over the GENERATED tokens
    mid1 = cache_resids(model, inp)["mid1"]                 # (bs,L,64) reproduces gen-time resid
    lg = model(inp).clone(); lg[..., BOS] = float("-inf")
    p = torch.softmax(lg, -1)[..., :D].cpu()               # (bs,L,D)

    c = counts_from_targets(gen).cpu()                     # (bs,L,D) exclusive running counts
    depleted = (c >= n)                                    # remaining == 0
    # (A) count tracking: probe vs true, MAE over digits, per position (skip t=0)
    pred = probe.predict(mid1.reshape(-1, 64).numpy()).reshape(bs, L, D)
    mae = np.abs(pred - c.numpy()).mean(-1)                # (bs,L)
    mae_pos = mae[:, 1:].mean(0)                           # (L-1,)
    # (B) leak: prob mass on depleted digits, per position
    leak = (p * depleted.float()).sum(-1).numpy()         # (bs,L)
    # invalid move at step i = the emitted digit was already depleted
    emit_depleted = depleted.gather(-1, gen.cpu()[..., None]).squeeze(-1).numpy()  # (bs,L) bool
    valid_so_far = (np.cumsum(emit_depleted, 1) - emit_depleted) == 0              # before step i
    # eps(i) on still-valid rollouts; empirical survival = fraction valid through i
    eps = np.array([leak[valid_so_far[:, i], i].mean() if valid_so_far[:, i].any() else 0.0 for i in range(L)])
    emp_surv = valid_so_far.mean(0)                        # fraction valid entering step i
    pred_surv = np.concatenate([[1.0], np.cumprod(1 - eps)])[:L]
    final_valid = (~emit_depleted).all(1).mean()
    # floor: mean prob on a depleted digit (per depleted slot), late positions
    late = c[:, 1:] >= n
    floor = (p[:, 1:][late]).mean().item() if late.any() else 0.0
    return dict(mae_pos=mae_pos, leak=leak[:, 1:].mean(0), eps=eps, emp_surv=emp_surv,
                pred_surv=pred_surv, final_valid=float(final_valid), floor=floor, n=n)


def main():
    t0 = time.time()
    Ls = [50, 100, 200, 300]
    res = {}
    for L in Ls:
        model, _ = load_model(CKDIR / f"L{L}.pt", device)
        probe = fit_count_probe(model, L)
        res[L] = {"sample": analyse(model, probe, L, rollout(model, L, "sample"))}
        if L == 50:
            res[L]["greedy"] = analyse(model, probe, L, rollout(model, L, "greedy"))
        s = res[L]["sample"]
        print(f"L={L:3d} sample valid={s['final_valid']:.3f} (pred {s['pred_surv'][-1]:.3f})  "
              f"depleted-digit prob floor={s['floor']:.4f}", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    g, sp = res[50]["greedy"], res[50]["sample"]
    tf_ref = None
    # (A) count tracking MAE on own rollouts (L=50)
    ax[0].plot(range(1, 50), g["mae_pos"], color="#3f7fb0", lw=2, label="greedy rollout")
    ax[0].plot(range(1, 50), sp["mae_pos"], color="#c0464b", lw=2, label="sampled rollout")
    ax[0].axhline(0.5, ls=":", c="gray", label="±0.5 count")
    ax[0].set(xlabel="generated position t", ylabel="probe count error (MAE, digits)",
              title="A) count tracking on the model's OWN rollouts (L=50)\n(stays small => counting is robust off-distribution)")
    ax[0].legend(fontsize=8)
    # (B) leak at the converter (L=50)
    ax[1].plot(range(1, 50), g["leak"], color="#3f7fb0", lw=2, label="greedy")
    ax[1].plot(range(1, 50), sp["leak"], color="#c0464b", lw=2, label="sampled")
    ax[1].set(xlabel="generated position t", ylabel="P(mass on depleted digits)  = eps(t)",
              title=f"B) leak onto exhausted digits (L=50)\nBayes=0; finite MLP floor ~{sp['floor']:.3f}")
    ax[1].legend(fontsize=8)
    # (C) survival accounting across lengths (sampling)
    cols = {50: "#3f7fb0", 100: "#2f8f4f", 200: "#e09020", 300: "#c0464b"}
    for L in Ls:
        s = res[L]["sample"]
        ax[2].plot(range(L), s["emp_surv"], color=cols[L], lw=2, label=f"L={L} empirical")
        ax[2].plot(range(L), s["pred_surv"], color=cols[L], lw=1, ls="--")
    ax[2].set(xlabel="generated position t", ylabel="fraction of rollouts still valid",
              title="C) survival: empirical (solid) vs prod(1-eps) (dashed)\nleak accounts for the length-dependent validity loss",
              ylim=(0, 1.02))
    ax[2].legend(fontsize=8)
    fig.suptitle("Phase 5: counting is robust on own rollouts; the (1-eps)^L erosion is the converter's depletion floor under sampling")
    fig.tight_layout(); fig.savefig(OUT / "phase5_decode.png", dpi=140); plt.close(fig)
    print(f"saved phase5_decode.png\nPHASE5 DONE ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
