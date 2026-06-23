"""Sequence-length sweep: train identical models at L in {50,100,200,250,500,1000}.

No changes other than L (and max_len bumped to 1024, required for L>511 and
functionally identical for shorter L). For each L: train 10k steps, save the best
checkpoint + the training history, then evaluate teacher-forced CE gap and
free-running decoding validity (greedy and ancestral T=1).
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_counting.train.trainer import (Train, TrainingArgs, ModelArgs,
                                        sample_batch)
from llm_counting.model.model import Transformer

SWEEP = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_sweep")
SWEEP.mkdir(parents=True, exist_ok=True)
LS = [50, 100, 200, 250, 500]
device = "cuda"; D = 10; VOCAB = 11


def load_best(L):
    ck = torch.load(SWEEP / f"L{L}/best.pt", map_location=device, weights_only=False)
    cfg = dict(ck["model_args"]); cfg.pop("VOCAB", None)
    m = Transformer(**cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def gen_validity(model, L, n, mode, B, T=1.0):
    bos = model.out.out_features - 1
    seq = torch.full((B, 1), bos, dtype=torch.long, device=device)
    for _ in range(L):
        lg = model(seq)[:, -1, :].clone(); lg[:, bos] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else \
              torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    gen = seq[:, 1:]
    cnt = torch.zeros(B, D, dtype=torch.long, device=device).scatter_add_(1, gen, torch.ones_like(gen))
    return (cnt == n).all(1).float().mean().item()


@torch.no_grad()
def tf_ce(model, L, B, nb=10):
    tot = 0.0
    for _ in range(nb):
        x, y = sample_batch(B, L, device=device)
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    return tot / nb


def main():
    rp = SWEEP / "results.json"
    results = {int(k): v for k, v in json.load(open(rp)).items()} if rp.exists() else {}
    for L in LS:
        if (SWEEP / f"L{L}/history.npz").exists() and L in results:
            print(f"skip L={L} (already done)", flush=True); continue
        t0 = time.time()
        targs = TrainingArgs(batch_size=256, num_steps=10000, seq_length=L, device=device,
                             lr=3e-4, val_every=200, log_every=50, weight_decay=0.01,
                             checkpoint_every_step=10_000_000, save_checkpoints=True, probe_size=4)
        margs = ModelArgs(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                          causal=True, max_len=1024, num_blocks=2, VOCAB=11)
        tr = Train(targs, margs); tr.ckpt_dir = SWEEP / f"L{L}"; tr.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== L={L}  n={L//D}  floor={tr.entropy_floor:.4f} =====", flush=True)
        tr.training_loop()

        g = [x if x is not None else np.nan for x in tr.history["gap"]]
        np.savez(SWEEP / f"L{L}/history.npz",
                 train_step=np.array(tr.history["train_step"]), train_loss=np.array(tr.history["train_loss"]),
                 val_step=np.array(tr.history["val_step"]), val_ce=np.array(tr.history["val_ce"]),
                 gap=np.array(g), floor=tr.entropy_floor, L=L, n=L // D)

        m = load_best(L); n = L // D; Be = 256 if L <= 500 else 128
        ce = tf_ce(m, L, Be)
        res = dict(L=L, n=n, floor=tr.entropy_floor, best_val_ce=tr.best_val,
                   tf_ce=ce, gap=ce - tr.entropy_floor,
                   greedy_valid=gen_validity(m, L, n, "greedy", Be),
                   sample_valid=gen_validity(m, L, n, "sample", Be, 1.0),
                   minutes=(time.time() - t0) / 60)
        results[L] = res
        print(f"L={L}: gap={res['gap']:.4f}  greedy_valid={res['greedy_valid']:.3f}  "
              f"sample_valid={res['sample_valid']:.3f}  ({res['minutes']:.1f} min)", flush=True)
        json.dump(results, open(SWEEP / "results.json", "w"), indent=2)
        del m, tr; torch.cuda.empty_cache()
    print("\nSWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
