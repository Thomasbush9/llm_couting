"""Decompose the parity feature across the four block-0 heads.

A) OV: project each head's per-digit write vectors onto the parity axis, and their SUM
   -> do the heads jointly map odd-tokens to one pole and even-tokens to the other?
B) attention: does any head attend preferentially to odd vs even key tokens (per key),
   or do they attend broadly and let the OV sort parity?
C) necessity: ablate each block-0 head, refit the regime probe at mid0 -> redundancy.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from scripts.probing_snippets import cache_resids
from scripts.run_parity import sample_batch_mix, ODDS, EVENS, L, D

device = "cuda"; H = 4
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def load():
    ck = torch.load("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_parity/parity.pt",
                    map_location=device, weights_only=False)
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    m.load_state_dict(ck["model"]); m.eval(); return m


def parity_axis(model, point="mid0", bs=512, tmin=20):
    x, y, isODD = sample_batch_mix(bs, device)
    R = cache_resids(model, x)[point][:, tmin:, :].reshape(-1, 64).numpy()
    lab = isODD[:, None].expand(-1, L - tmin).reshape(-1).cpu().numpy()
    clf = LogisticRegression(max_iter=1000).fit(R, lab)
    w = clf.coef_[0]; return w / np.linalg.norm(w), clf


def ov_write(model, h):
    blk = model.blocks[0]; mha = blk.mha; dk = mha.lin_v.out_features // H
    with torch.no_grad():
        e = blk.norm1(model.emb.weight[:10])
        v = mha.lin_v(e).view(10, H, dk)
        vm = torch.zeros_like(v); vm[:, h] = v[:, h]
        return (vm.reshape(10, H * dk) @ mha.out.weight.T).cpu().numpy()   # (10,64)


@torch.no_grad()
def attn_by_parity(model, bs=256):
    x, y, _ = sample_batch_mix(bs, device)
    model(x)
    A = model.blocks[0].mha.attn                       # (B,H,L,L)
    tok = x                                            # keys (incl BOS at idx0)
    odd = ((tok % 2 == 1) & (tok < 10)).float()
    even = ((tok % 2 == 0) & (tok < 10)).float()
    q0 = L // 2                                         # later queries: most keys available
    def per_key(mask):
        num = (A[:, :, q0:, :] * mask[:, None, None, :]).sum(-1)         # (B,H,Q)
        den = mask.sum(1)[:, None, None].clamp(min=1)
        return (num / den)[:, :, :].mean(dim=(0, 2)).cpu().numpy()       # (H,)
    return per_key(odd), per_key(even)


def main():
    model = load()
    wdir, _ = parity_axis(model, "mid0")

    # A) OV per head + sum, projected on parity axis
    order = EVENS + ODDS
    rows = []
    for h in range(H):
        rows.append(ov_write(model, h) @ wdir)
    rows = np.array(rows)                                # (H,10)
    total = rows.sum(0)
    M = np.vstack([rows, total])[:, order]               # (H+1, 10) reordered evens|odds

    # B) attention per odd/even key
    a_odd, a_even = attn_by_parity(model)
    print("attn per odd-key:", np.round(a_odd, 4), "\nattn per even-key:", np.round(a_even, 4))

    # C) per-head necessity: ablate each block-0 head, regime decodability at mid0
    def regime_acc():
        Xtr, ytr = _eval_xy(model); Xte, yte = _eval_xy(model)
        return LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    base_acc = regime_acc()
    accs = []
    for h in range(H):
        model.blocks[0].mha.ablate_heads = [h]
        accs.append(regime_acc())
        model.blocks[0].mha.ablate_heads = []
    print("baseline regime decodability (mid0):", round(100 * base_acc, 1))
    print("regime decodability ablating each block-0 head:", [round(100 * a, 1) for a in accs])

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    vlim = np.abs(M).max()
    im = ax[0].imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax[0].set_yticks(range(H + 1)); ax[0].set_yticklabels([f"head{h}" for h in range(H)] + ["SUM"])
    ax[0].set_xticks(range(10)); ax[0].set_xticklabels(order); ax[0].axvline(4.5, color="k", lw=1)
    ax[0].axhline(H - 0.5, color="k", lw=1.5)
    ax[0].set(title="A) OV write · parity axis\n(cols: evens | odds)", xlabel="digit")
    fig.colorbar(im, ax=ax[0])

    xh = np.arange(H)
    ax[1].bar(xh - 0.2, a_odd, 0.4, color="#c0464b", label="per odd key")
    ax[1].bar(xh + 0.2, a_even, 0.4, color="#3f7fb0", label="per even key")
    ax[1].set_xticks(xh); ax[1].set_xticklabels([f"h{h}" for h in range(H)])
    ax[1].set(title="B) block-0 attention by key parity", ylabel="mean attention weight / key")
    ax[1].legend()

    ax[2].bar(xh, [100 * a for a in accs], color="#2f8f4f")
    ax[2].axhline(100 * base_acc, ls="--", c="gray", label="baseline (no ablation)")
    ax[2].set_xticks(xh); ax[2].set_xticklabels([f"h{h}" for h in range(H)])
    ax[2].set(title="C) regime decodability when head ablated", ylabel="%", ylim=(45, 102))
    ax[2].legend()
    fig.suptitle("Parity feature decomposed across the four block-0 heads")
    fig.tight_layout(); fig.savefig(OUT / "parity_heads.png", dpi=110); plt.close(fig)
    print("SUM row (evens|odds):", np.round(total[order], 2))
    print("saved parity_heads.png")


def _eval_xy(model, bs=512, tmin=20):
    x, y, isODD = sample_batch_mix(bs, device)
    R = cache_resids(model, x)["mid0"][:, tmin:, :].reshape(-1, 64).numpy()
    lab = isODD[:, None].expand(-1, L - tmin).reshape(-1).cpu().numpy()
    return R, lab


if __name__ == "__main__":
    main()
