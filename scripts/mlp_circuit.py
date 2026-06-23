"""Does the MLP convert counts -> probability?

The Bayes map for this task is P(next=d) = r_d / R with r_d = n - c_d (remaining)
and R = sum_d r_d. A softmax produces that ratio exactly when the logits are
log r_d (+const). Turning a *linear* count code into *log*-counts is a nonlinear
step -- the natural suspect is the MLP. This script tests that.

NOTE: as implemented, MLP.forward does `out = self.output(x)` (ignoring
self.input) with no activation, so each MLP block is a single AFFINE map and the
network has no MLP nonlinearity at all. The experiments below quantify what that
costs and where the count->prob conversion actually happens.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_counting.model.model import Transformer
from llm_counting.train.trainer import sample_batch, true_conditional, entropy_floor

D = 10


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["model_args"]); cfg.pop("VOCAB", None)
    model = Transformer(**cfg).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck


def counts_from_targets(targets):
    oh = F.one_hot(targets, num_classes=D).float()
    return torch.cumsum(oh, dim=1) - oh


class MLPAblation:
    """Context manager: zero every block's MLP contribution to the residual."""
    def __init__(self, model):
        self.model = model; self.handles = []

    def __enter__(self):
        def zero(m, i, o):
            return torch.zeros_like(o)
        self.handles = [blk.mlp.register_forward_hook(zero) for blk in self.model.blocks]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def confirm_linear(model, device):
    mlp = model.blocks[0].mlp
    x = torch.randn(8, 64, device=device)
    uses_x_only = torch.allclose(mlp(x), mlp.output(x), atol=1e-6)
    # affine check: mlp(a x) - mlp(0) == a (mlp(x) - mlp(0))
    z = torch.zeros_like(x); b = mlp(z)
    linear = torch.allclose(mlp(3 * x) - b, 3 * (mlp(x) - b), atol=1e-4)
    print(f"  MLP(x) == output(x) (input layer dead): {uses_x_only}")
    print(f"  MLP is affine (no nonlinearity):        {linear}")


@torch.no_grad()
def ce_and_calibration(model, L, device, n_batches=20, subsample=4000, seed_pts=None):
    floor = entropy_floor(seq_len=L, batch_size=256, device=str(device))
    out = {}
    for tag, ablate in [("full", False), ("no-MLP", True)]:
        ctx = MLPAblation(model) if ablate else None
        if ctx: ctx.__enter__()
        tot, pm, pt = 0.0, [], []
        for _ in range(n_batches):
            inp, tgt = sample_batch(256, L, device=device)
            logits = model(inp)
            V = logits.size(-1)
            tot += F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1)).item()
            masked = logits.clone(); masked[..., -1] = float("-inf")
            pm.append(torch.softmax(masked, -1)[..., :D].reshape(-1).cpu())
            pt.append(true_conditional(tgt, VOCAB=D + 1).reshape(-1).cpu())
        if ctx: ctx.__exit__()
        out[tag] = dict(ce=tot / n_batches,
                        pm=torch.cat(pm).numpy(), pt=torch.cat(pt).numpy())
        print(f"  {tag:7s} CE={out[tag]['ce']:.4f}  gap={out[tag]['ce']-floor:.4f}")
    out["floor"] = floor
    return out


def plot_calibration(cal, out):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for tag, c in [("full", "#3f7fb0"), ("no-MLP", "#c0464b")]:
        pm, pt = cal[tag]["pm"], cal[tag]["pt"]
        idx = np.random.default_rng(0).choice(len(pm), size=min(4000, len(pm)), replace=False)
        ax[0].scatter(pt[idx], pm[idx], s=4, alpha=0.25, color=c, label=tag)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(xlabel="Bayes P(next=d) = r_d/R", ylabel="model P(next=d)",
              title="calibration (next-token prob)")
    ax[0].legend()

    tags = ["full", "no-MLP"]
    ax[1].bar(tags, [cal[t]["ce"] for t in tags], color=["#3f7fb0", "#c0464b"])
    ax[1].axhline(cal["floor"], ls="--", c="gray", label=f"Bayes floor {cal['floor']:.3f}")
    ax[1].set(ylabel="val cross-entropy (nats)", title="MLP ablation -> CE")
    ax[1].set_ylim(cal["floor"] - 0.02, max(cal[t]["ce"] for t in tags) + 0.05)
    ax[1].legend()
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


@torch.no_grad()
def count_to_logit(model, L, device, out, n_batches=12):
    """Test logit_d ~ log(r_d). Softmax is shift-invariant, so the right comparison
    is per-position-centered: centered logit_d vs centered log(remaining), where the
    centering is over the digits still available (r>=1) at that position. Depleted
    digits (r=0) are handled separately (they should get a large negative logit)."""
    n = L // D

    def collect(ablate):
        xs, ys, dep = [], [], []        # centered log-r, centered logit, depleted logits
        ctx = MLPAblation(model) if ablate else None
        if ctx: ctx.__enter__()
        for _ in range(n_batches):
            inp, tgt = sample_batch(256, L, device=device)
            lg = model(inp)[..., :D].reshape(-1, D).cpu().numpy()          # (N,D) digit logits
            r  = (n - counts_from_targets(tgt)).clamp(min=0).reshape(-1, D).cpu().numpy()
            mask = r >= 1
            cnt = mask.sum(1, keepdims=True)
            ok = (cnt[:, 0] >= 2)                                          # need >=2 to center
            logr = np.log(np.where(mask, r, 1.0))
            lr_c = logr - (logr * mask).sum(1, keepdims=True) / cnt
            lg_c = lg   - (lg   * mask).sum(1, keepdims=True) / cnt
            xs.append(lr_c[ok][mask[ok]]); ys.append(lg_c[ok][mask[ok]])
            # depleted digits: model logit relative to surviving-mean
            lg_dep = lg - (lg * mask).sum(1, keepdims=True) / cnt
            dep.append(lg_dep[(~mask) & (cnt >= 1)])
        if ctx: ctx.__exit__()
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(dep)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tag, ablate) in zip(axes, [("full", False), ("no-MLP", True)]):
        x, y, dep = collect(ablate)
        slope, intercept = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (slope * x + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {tag:7s}: centered logit = {slope:.2f}*log(r_d) + {intercept:.2f}"
              f"   R2={r2:.3f}   depleted-digit mean logit={dep.mean():.2f}")
        ax.hist2d(x, y, bins=60, cmap="Blues", cmin=1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, "r-", lw=2, label=f"fit slope={slope:.2f}, R²={r2:.2f}")
        ax.plot(xs, xs, "k--", lw=1, label="ideal (slope 1 = exact log)")
        ax.set(title=f"{tag}: count -> logit", xlabel="centered log(remaining r_d)",
               ylabel="centered model logit_d")
        ax.legend(loc="upper left")
    fig.suptitle("does the network map counts to log-remaining? (softmax then gives r_d/R)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(args.ckpt, device)
    L = ck["args"]["seq_length"]
    print(f"checkpoint step={ck['step']}  L={L}  n={L//D}\n")

    print("[1] is the MLP nonlinear?")
    confirm_linear(model, device)

    print("\n[2] MLP ablation: CE + calibration")
    cal = ce_and_calibration(model, L, device)
    plot_calibration(cal, outdir / "mlp_ablation.png")

    print("\n[3] count -> logit functional form")
    count_to_logit(model, L, device, outdir / "count_to_logit.png")


if __name__ == "__main__":
    main()
