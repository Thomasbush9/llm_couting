"""Count vs regime: do the per-digit count code and the parity-regime axis share
directions, or are they orthogonal subspaces? (parity model, residual at mid1)

We fit (a) per-digit count read-out directions and (b) the regime (parity) axis, then ask:
 - is the regime axis aligned with the *parity contrast* of the count directions?
 - what fraction of the regime axis lies inside the 10-d count subspace?
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge, LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from scripts.probing_snippets import cache_resids
from scripts.run_parity import sample_batch_mix, ODDS, EVENS, L, D

device = "cuda"
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def load():
    ck = torch.load("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_parity/parity.pt",
                    map_location=device, weights_only=False)
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    m.load_state_dict(ck["model"]); m.eval(); return m


def main():
    model = load()
    x, y, isODD = sample_batch_mix(512, device)
    P = cache_resids(model, x)
    X = P["mid1"].reshape(-1, 64).numpy()
    oh = F.one_hot(y, D).float(); c = (torch.cumsum(oh, 1) - oh).reshape(-1, D).cpu().numpy()
    Rc = Ridge(1.0).fit(X, c).coef_                                  # (10,64) count directions

    Xr = P["mid1"][:, 20:, :].reshape(-1, 64).numpy()
    lab = isODD[:, None].expand(-1, L - 20).reshape(-1).cpu().numpy()
    Rr = LogisticRegression(max_iter=1000).fit(Xr, lab).coef_[0]
    Rr = Rr / np.linalg.norm(Rr)                                     # regime (parity) axis

    Rcn = Rc / np.linalg.norm(Rc, axis=1, keepdims=True)
    cos_d = Rcn @ Rr                                                 # cos(count_d, regime)
    parity_contrast = Rc[ODDS].mean(0) - Rc[EVENS].mean(0)
    pc = parity_contrast / np.linalg.norm(parity_contrast)
    cos_pc = float(pc @ Rr)
    a, *_ = np.linalg.lstsq(Rc.T, Rr, rcond=None)                    # best Rr ≈ Rc.T @ a
    frac_in_count = float(1 - np.linalg.norm(Rr - Rc.T @ a) ** 2 / np.linalg.norm(Rr) ** 2)
    # how aligned is each count direction with the parity contrast (the shared mode)?
    cos_d_pc = Rcn @ pc

    print(f"cos(regime axis, parity-contrast of counts) = {cos_pc:.3f}")
    print(f"fraction of regime axis inside the 10-d count subspace = {frac_in_count:.3f}")
    print("cos(count_d, regime) per digit:", {d: round(float(cos_d[d]), 2) for d in range(D)})

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    col = ["#c0464b" if d in ODDS else "#3f7fb0" for d in range(D)]
    ax[0].bar(range(D), cos_d, color=col)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="digit", ylabel="cosine(count direction, regime axis)", xticks=range(D),
              title="A) each digit's count direction vs the regime axis\n(red=odd, blue=even)")

    ax[1].bar([0, 1], [cos_pc, frac_in_count], color=["#2f8f4f", "#9467bd"])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["cos(regime,\nparity-contrast\nof counts)",
                                                     "frac. of regime axis\nin count subspace"])
    ax[1].set(ylim=(0, 1.05), title="B) regime axis lives in the count subspace")
    for i, v in enumerate([cos_pc, frac_in_count]):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.suptitle("Count code vs parity-regime axis — shared, not orthogonal")
    fig.tight_layout(); fig.savefig(OUT / "count_vs_regime.png", dpi=120); plt.close(fig)
    print("saved count_vs_regime.png")


if __name__ == "__main__":
    main()
