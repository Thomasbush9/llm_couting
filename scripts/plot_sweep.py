"""Plot the sequence-length sweep: training curves + decoding performance vs L."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_sweep")
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
LS = [50, 100, 200, 250, 500, 1000]


def main():
    hist = {}
    for L in LS:
        p = SWEEP / f"L{L}/history.npz"
        if p.exists():
            hist[L] = dict(np.load(p))
    res = json.load(open(SWEEP / "results.json")) if (SWEEP / "results.json").exists() else {}
    res = {int(k): v for k, v in res.items()}
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(LS)))
    col = {L: cmap[i] for i, L in enumerate(LS)}

    # ---- Figure 1: training convergence (gap to Bayes floor vs step) ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for L in LS:
        if L not in hist: continue
        h = hist[L]
        ax[0].plot(h["val_step"], h["val_ce"], color=col[L], lw=1.8, label=f"L={L} (n={L//10})")
        ax[0].axhline(float(h["floor"]), color=col[L], ls=":", lw=1)
        gap = h["gap"].copy(); gap[gap <= 0] = np.nan
        ax[1].plot(h["val_step"], gap, color=col[L], lw=1.8, label=f"L={L}")
    ax[0].set(xlabel="step", ylabel="val cross-entropy (nats)",
              title="validation CE (dotted = per-L Bayes floor)")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="step", ylabel="val CE − Bayes floor (nats)", yscale="log",
              title="gap to Bayes-optimal vs training step")
    ax[1].legend(fontsize=8)
    fig.suptitle("Sequence-length sweep — training convergence")
    fig.tight_layout(); fig.savefig(OUT / "sweep_training_curves.png", dpi=150); plt.close(fig)
    print("saved sweep_training_curves.png")

    # ---- Figure 2: performance vs L ----
    if res:
        Ls = sorted(res)
        gap = [res[L]["gap"] for L in Ls]
        gv = [100 * res[L]["greedy_valid"] for L in Ls]
        sv = [100 * res[L]["sample_valid"] for L in Ls]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(Ls, gap, "o-", lw=2, color="#c0464b")
        ax[0].set(xlabel="sequence length L", ylabel="final gap to Bayes floor (nats)",
                  xscale="log", yscale="log", title="teacher-forced gap vs L")
        ax[0].set_xticks(Ls); ax[0].set_xticklabels(Ls)
        ax[1].plot(Ls, gv, "o-", lw=2, label="greedy", color="#3f7fb0")
        ax[1].plot(Ls, sv, "s-", lw=2, label="ancestral T=1", color="#2f8f4f")
        ax[1].set(xlabel="sequence length L", ylabel="valid balanced sequences (%)",
                  xscale="log", title="free-running decoding validity vs L", ylim=(-3, 103))
        ax[1].set_xticks(Ls); ax[1].set_xticklabels(Ls); ax[1].legend()
        fig.suptitle("Sequence-length sweep — performance")
        fig.tight_layout(); fig.savefig(OUT / "sweep_performance.png", dpi=150); plt.close(fig)
        print("saved sweep_performance.png")
        print("\nL     n    floor    gap      greedy%  sample%")
        for L in Ls:
            r = res[L]
            print(f"{L:<5} {r['n']:<4} {r['floor']:.3f}   {r['gap']:.4f}  "
                  f"{100*r['greedy_valid']:5.1f}   {100*r['sample_valid']:5.1f}")


if __name__ == "__main__":
    main()
