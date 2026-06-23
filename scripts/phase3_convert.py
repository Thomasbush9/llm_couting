"""Phase 3 - how counts become probabilities (the count -> log-remaining converter).

Bayes says P(next=d) = r_d/R with r_d = n - c_d.  Softmax-inverting: the logit for
digit d must be  logit_d = log(n - c_d) + (a per-position constant, shared across d).
So the network's job after counting is a per-digit NONLINEAR map  c_d -> log(n - c_d).

We test, on the corrected near-Bayes L=50 model:

  (A) Within-position alignment: regress the model's (digit-centered) logits on the
      (digit-centered) true log-remaining log(n - c_d). slope ~1, R2 ~1 => the model
      literally outputs log-remaining. Compare clean vs MLP-ablated.

  (B) MLP necessity (causal): ablate block-0 / block-1 / both MLP contributions and
      measure KL(true||model) and the log-remaining slope. The nonlinearity is what
      bends a linear count read-out into log(n - c).

  (C) The map itself: bin by c_d and show mean model logit_d tracks log(n - c_d),
      including the hard depletion regime c_d -> n (logit -> -inf).

Outputs: phase3_convert.png  (+ printed table).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.train.trainer import sample_batch, true_conditional
from scripts.probing_snippets import load_model, counts_from_targets

device = "cuda"; D = 10
CK = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50/best.pt"
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def ablate_mlp(model, which):
    """Return handles that zero the residual contribution of the named MLPs."""
    z = lambda m, i, o: torch.zeros_like(o)
    hs = []
    if "b0" in which: hs.append(model.blocks[0].mlp.register_forward_hook(z))
    if "b1" in which: hs.append(model.blocks[1].mlp.register_forward_hook(z))
    return hs


@torch.no_grad()
def run(model, x, which=()):
    hs = ablate_mlp(model, which)
    lg = model(x).clone()
    for h in hs: h.remove()
    return lg


def centered_within_pos(arr, valid):
    """arr,valid: (N,D). Subtract per-row mean over VALID digits; return flattened valid."""
    a = arr.astype(float).copy(); a[~valid] = np.nan
    a = a - np.nanmean(a, axis=1, keepdims=True)
    return a[valid]


def align(model, x, y, which, n):
    lg = run(model, x, which)[..., :D].cpu().numpy().reshape(-1, D)
    c = counts_from_targets(y).cpu().numpy().reshape(-1, D)
    valid = c < n                                          # exclude depleted (log 0)
    lr = np.where(valid, np.log(np.clip(n - c, 1e-9, None)), np.nan)
    X = centered_within_pos(lr, valid); Y = centered_within_pos(lg, valid)
    slope = float((X * Y).sum() / (X * X).sum())
    r2 = float(1 - ((Y - slope * X) ** 2).sum() / (Y ** 2).sum())
    # KL to Bayes
    lg2 = run(model, x, which).clone(); lg2[..., -1] = float("-inf")
    pm = torch.softmax(lg2, -1)[..., :D].cpu(); pt = true_conditional(y.cpu(), VOCAB=D + 1)
    kl = (torch.xlogy(pt, pt) - torch.xlogy(pt, pm.clamp(min=1e-12))).sum(-1)[:, 1:].mean().item()
    return slope, r2, kl, (X, Y)


def main():
    t0 = time.time()
    model, ck = load_model(CK, device)
    L = ck["args"]["seq_length"]; n = L // D
    x, y = sample_batch(512, L, device=device)
    conds = [("clean", ()), ("no b1-mlp", ("b1",)), ("no b0-mlp", ("b0",)), ("no MLPs", ("b0", "b1"))]
    res = {}
    print(f"L={L} n={n}")
    for name, w in conds:
        s, r2, kl, sc = align(model, x, y, w, n)
        res[name] = dict(slope=s, r2=r2, kl=kl, sc=sc)
        print(f"  {name:<10} log-remaining slope={s:.3f}  R2={r2:.3f}  KL={kl:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # A) scatter clean vs no-MLP
    for name, col in [("clean", "#2f8f4f"), ("no MLPs", "#c0464b")]:
        X, Y = res[name]["sc"]; idx = np.random.default_rng(0).choice(len(X), size=4000, replace=False)
        ax[0].scatter(X[idx], Y[idx], s=4, alpha=0.25, color=col,
                      label=f"{name} (slope {res[name]['slope']:.2f}, R2 {res[name]['r2']:.2f})")
    lim = [-2, 2]; ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel="true centered log(n - c_d)", ylabel="model centered logit_d",
              xlim=lim, ylim=[-3, 3], title="A) model logits = log-remaining?"); ax[0].legend(fontsize=8)
    # B) KL bars
    names = [c[0] for c in conds]; xp = np.arange(len(names))
    ax[1].bar(xp, [res[m]["kl"] for m in names], color=["#3f7fb0", "#9467bd", "#e09020", "#c0464b"])
    ax[1].set_xticks(xp); ax[1].set_xticklabels(names, rotation=15)
    ax[1].set(ylabel="KL(true||model)", title="B) MLP necessity for the conversion")
    for i, m in enumerate(names): ax[1].text(i, res[m]["kl"], f"{res[m]['kl']:.3f}", ha="center", va="bottom", fontsize=8)
    # C) the count->logit map for one digit, binned, clean vs no-MLP
    dd = 6
    lgc = run(model, x, ())[..., :D].cpu().numpy().reshape(-1, D)
    lgn = run(model, x, ("b0", "b1"))[..., :D].cpu().numpy().reshape(-1, D)
    c = counts_from_targets(y).cpu().numpy().reshape(-1, D)
    rowmean_c = lgc.mean(1, keepdims=True); rowmean_n = lgn.mean(1, keepdims=True)
    cc = c[:, dd]
    for lab, arr, rm, col in [("clean", lgc, rowmean_c, "#2f8f4f"), ("no MLPs", lgn, rowmean_n, "#c0464b")]:
        ys = (arr[:, dd:dd + 1] - rm).ravel()
        mean_by_c = [ys[cc == k].mean() for k in range(n + 1)]
        ax[2].plot(range(n + 1), mean_by_c, "o-", color=col, label=lab)
    ax[2].plot(range(n), [np.log(n - k) - np.log(np.maximum(n - np.arange(n), 1e-9)).mean() for k in range(n)],
               "k--", lw=1.5, label="log(n - c) (Bayes)")
    ax[2].set(xlabel=f"count of digit {dd}", ylabel="centered logit", title="C) count -> logit map (digit 6)")
    ax[2].legend(fontsize=8)
    fig.suptitle("Phase 3: the MLP converts count -> log-remaining, so softmax outputs Bayes r_d/R")
    fig.tight_layout(); fig.savefig(OUT / "phase3_convert.png", dpi=140); plt.close(fig)
    print(f"saved phase3_convert.png\nPHASE3 DONE ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
