"""Trace the parity -> output pathway in the parity-mixture model.

A) Direct logit attribution (DLA): decompose the parity decision at the logits
   (odd-digit logit mass minus even) into each residual component's direct write
   (emb, block-0 attn, block-0 mlp, block-1 attn, block-1 mlp). Which component
   actually writes "suppress the minority parity" to the output?

B) Causal patching: run an odd-heavy batch, overwrite one component's output with
   the activation from an even-heavy batch, and measure how much the output parity
   flips. Which component is causally responsible for carrying the regime to the logits?
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from scripts.probing_snippets import cache_resids
from scripts.run_parity import sample_batch_mix, ODDS, EVENS, L, D

device = "cuda"; TMIN = 20
COMPS = ["emb", "b0-attn", "b0-mlp", "b1-attn", "b1-mlp"]
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def load():
    ck = torch.load("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_parity/parity.pt",
                    map_location=device, weights_only=False)
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    m.load_state_dict(ck["model"]); m.eval(); return m


def parity_mass(logits):
    """P(next in odd) - P(next in even), pre-masked BOS, averaged over positions>=TMIN."""
    m = logits.clone(); m[..., 10] = float("-inf")
    P = torch.softmax(m, -1)
    return (P[..., ODDS].sum(-1) - P[..., EVENS].sum(-1))[:, TMIN:].mean().item()


@torch.no_grad()
def dla(model):
    # parity direction in unembedding space
    W = model.out.weight.detach().cpu()
    u = W[ODDS].mean(0) - W[EVENS].mean(0)                 # (64,)
    gamma = model.norm.weight.detach().cpu()
    out = {}
    for tag, pODD in [("odd-heavy", 1.0), ("even-heavy", 0.0)]:
        x, y, _ = sample_batch_mix(512, device, pODD=pODD)
        p = cache_resids(model, x)
        deltas = {"emb": p["emb"], "b0-attn": p["mid0"] - p["emb"], "b0-mlp": p["post0"] - p["mid0"],
                  "b1-attn": p["mid1"] - p["post0"], "b1-mlp": p["post1"] - p["mid1"]}
        sigma = (p["post1"].var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        scale = (u * gamma) / sigma                        # (B,L,64)
        out[tag] = {c: (scale * deltas[c]).sum(-1)[:, TMIN:].mean().item() for c in COMPS}
    return out


@torch.no_grad()
def patch(model):
    mods = {"b0-attn": model.blocks[0].mha, "b0-mlp": model.blocks[0].mlp,
            "b1-attn": model.blocks[1].mha, "b1-mlp": model.blocks[1].mlp}
    xo, _, _ = sample_batch_mix(512, device, pODD=1.0)     # clean (odd-heavy)
    xe, _, _ = sample_batch_mix(512, device, pODD=0.0)     # corrupt source (even-heavy)
    base_odd = parity_mass(model(xo)); base_even = parity_mass(model(xe))
    # cache even-heavy component outputs
    store = {}
    hs = [m.register_forward_hook(lambda mod, i, o, n=n: store.__setitem__(n, o.detach())) for n, m in mods.items()]
    model(xe)
    for h in hs: h.remove()
    flips = {}
    for n, m in mods.items():
        h = m.register_forward_hook(lambda mod, i, o, n=n: store[n])
        patched = parity_mass(model(xo))
        h.remove()
        flips[n] = (base_odd - patched) / (base_odd - base_even)   # 1 = fully flipped to even-regime
    return base_odd, base_even, flips


def main():
    model = load()
    A = dla(model); base_odd, base_even, flips = patch(model)
    print("clean parity mass: odd-heavy=%.3f even-heavy=%.3f" % (base_odd, base_even))
    print("DLA (odd-heavy):", {c: round(A["odd-heavy"][c], 3) for c in COMPS})
    print("patch flip fraction:", {c: round(flips[c], 3) for c in flips})

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(COMPS))
    ax[0].bar(x - 0.2, [A["odd-heavy"][c] for c in COMPS], 0.4, color="#c0464b", label="odd-heavy")
    ax[0].bar(x + 0.2, [A["even-heavy"][c] for c in COMPS], 0.4, color="#3f7fb0", label="even-heavy")
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels(COMPS, rotation=20)
    ax[0].set(ylabel="direct contribution to parity logit\n(odd mass − even mass)",
              title="A) which component writes the parity decision (DLA)"); ax[0].legend()

    pc = [c for c in COMPS if c != "emb"]
    ax[1].bar(range(len(pc)), [100 * flips[c] for c in pc], color="#2f8f4f")
    ax[1].axhline(100, ls=":", c="gray", label="full flip to even-regime")
    ax[1].set_xticks(range(len(pc))); ax[1].set_xticklabels(pc, rotation=20)
    ax[1].set(ylabel="output parity flip (%)",
              title="B) patch component from even-heavy → odd-heavy run"); ax[1].legend()
    fig.suptitle("Parity → output pathway: computed in block-0 attn, written to logits downstream")
    fig.tight_layout(); fig.savefig(OUT / "parity_pathway.png", dpi=120); plt.close(fig)
    print("saved parity_pathway.png")


if __name__ == "__main__":
    main()
