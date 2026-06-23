"""Length sweep on the CORRECTED (GELU) counting model.

For each L in a sweep: train to convergence, then measure
  - gap to the Bayes floor (teacher-forced optimality),
  - decoding validity: fraction of greedy / ancestral-T1 rollouts that are exactly
    balanced (each digit exactly n=L/D times),
  - the circuit is intact at this length: count R2 at mid0, and the count->log-remaining
    slope/R2 at the output (the Phase 1-3 signatures).

So we see not just "how well it performs at length L" but whether the SAME mechanism
(count via block-0 attn, MLP -> log-remaining) carries across lengths.

Outputs: phase_length_sweep.png + checkpoints_count50/L{L}.pt + sweep_results.json
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from llm_counting.train.trainer import sample_batch, true_conditional, entropy_floor
from scripts.probing_snippets import cache_resids, counts_from_targets

device = "cuda"; D = 10; VOCAB = 11
LSET = [20, 50, 100, 200, 300]
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50")
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
MA = dict(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
          causal=True, max_len=512, num_blocks=2)


@torch.no_grad()
def val_gap(model, L, floor, nb=20, bs=256):
    ce = 0.0
    for _ in range(nb):
        x, y = sample_batch(bs, L, device=device)
        ce += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    return ce / nb - floor


def train(L, steps, bs=256):
    floor = entropy_floor(seq_len=L, batch_size=256, device=device, n_batches=40)
    m = Transformer(**dict(MA, max_len=max(512, L + 2))).to(device)
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    best, best_state = float("inf"), None
    for s in range(steps):
        m.train(); x, y = sample_batch(bs, L, device=device)
        opt.zero_grad(); F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1)).backward(); opt.step()
        if s % 500 == 0:
            m.eval(); g = val_gap(m, L, floor, nb=8)
            if g < best: best = g; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    return m, floor


@torch.no_grad()
def decode_valid(model, L, mode, bs=512, T=1.0):
    n = L // D; bos = VOCAB - 1
    seq = torch.full((bs, 1), bos, dtype=torch.long, device=device)
    for _ in range(L):
        lg = model(seq)[:, -1, :].clone(); lg[:, bos] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    cnt = torch.zeros(bs, D, dtype=torch.long, device=device).scatter_add_(
        1, seq[:, 1:], torch.ones_like(seq[:, 1:]))
    return (cnt == n).all(1).float().mean().item()


@torch.no_grad()
def circuit_metrics(model, L):
    n = L // D
    xtr, ytr = sample_batch(256, L, device=device); xte, yte = sample_batch(256, L, device=device)
    Ptr, Pte = cache_resids(model, xtr), cache_resids(model, xte)
    Xtr = Ptr["mid0"][:, 1:].reshape(-1, 64).numpy(); Xte = Pte["mid0"][:, 1:].reshape(-1, 64).numpy()
    ctr = counts_from_targets(ytr).cpu()[:, 1:].reshape(-1, D).numpy()
    cte = counts_from_targets(yte).cpu()[:, 1:].reshape(-1, D).numpy()
    r2_count = r2_score(cte, Ridge(1.0).fit(Xtr, ctr).predict(Xte), multioutput="uniform_average")
    # log-remaining slope/R2 at output (drop position 0 to match cte)
    lg = model(xte)[:, 1:, :D].cpu().numpy().reshape(-1, D)
    c = cte; valid = c < n
    lr = np.where(valid, np.log(np.clip(n - c, 1e-9, None)), np.nan)
    def cen(a):
        a = a.astype(float).copy(); a[~valid] = np.nan
        return (a - np.nanmean(a, 1, keepdims=True))[valid]
    X, Y = cen(lr), cen(lg)
    slope = float((X * Y).sum() / (X * X).sum()); r2_lr = float(1 - ((Y - slope * X) ** 2).sum() / (Y ** 2).sum())
    return r2_count, slope, r2_lr


def main():
    t0 = time.time(); res = {}
    for L in LSET:
        steps = 12000 if L <= 100 else 16000
        m, floor = train(L, steps)
        gap = val_gap(m, L, floor, nb=40)
        gv, sv = decode_valid(m, L, "greedy"), decode_valid(m, L, "sample")
        r2c, slope, r2lr = circuit_metrics(m, L)
        torch.save({"step": steps, "model": m.state_dict(), "args": {"seq_length": L},
                    "model_args": dict(MA, max_len=max(512, L + 2), VOCAB=VOCAB)}, CK / f"L{L}.pt")
        res[L] = dict(floor=floor, gap=gap, greedy_valid=gv, sample_valid=sv,
                      r2_count=r2c, lr_slope=slope, lr_r2=r2lr)
        print(f"L={L:3d} gap={gap:.4f} greedy={gv:.3f} sample={sv:.3f} | "
              f"countR2={r2c:.3f} logrem_slope={slope:.3f} R2={r2lr:.3f}", flush=True)
    json.dump(res, open(CK / "sweep_results.json", "w"), indent=2)

    Ls = LSET
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(Ls, [res[L]["gap"] for L in Ls], "o-", color="#2f8f4f")
    ax[0].set(xlabel="L", ylabel="gap to Bayes floor (nats)", title="A) teacher-forced optimality vs length")
    ax[0].set_yscale("log")
    ax[1].plot(Ls, [100 * res[L]["greedy_valid"] for L in Ls], "o-", color="#3f7fb0", label="greedy")
    ax[1].plot(Ls, [100 * res[L]["sample_valid"] for L in Ls], "s-", color="#c0464b", label="ancestral T=1")
    ax[1].set(xlabel="L", ylabel="valid balanced rollouts (%)", title="B) decoding validity vs length", ylim=(-2, 103))
    ax[1].legend()
    ax[2].plot(Ls, [res[L]["r2_count"] for L in Ls], "o-", color="#c0464b", label="count R2 @ mid0")
    ax[2].plot(Ls, [res[L]["lr_r2"] for L in Ls], "s-", color="#2f8f4f", label="log-remaining R2")
    ax[2].plot(Ls, [res[L]["lr_slope"] for L in Ls], "^--", color="#9467bd", label="log-remaining slope")
    ax[2].axhline(1.0, ls=":", c="gray"); ax[2].set(xlabel="L", ylabel="value",
        title="C) same circuit at every length?", ylim=(0, 1.3)); ax[2].legend(fontsize=8)
    fig.suptitle("Length sweep (corrected GELU model): performance + circuit signatures")
    fig.tight_layout(); fig.savefig(OUT / "phase_length_sweep.png", dpi=140); plt.close(fig)
    print(f"saved phase_length_sweep.png\nSWEEP DONE ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
