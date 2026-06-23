"""Mechanistic circuit analysis of the parity-mixture model.

Q1 (localize): where in the residual stream does the parity regime become linearly
    decodable?  -> probe regime at emb/mid0/post0/mid1/post1/final.
Q2 (which heads): per-head ablation -> effect on task CE and on regime decodability.
Q3 (how): for the key head(s), do the per-digit OV write vectors separate odd vs even?
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
from scripts.run_parity import sample_batch_mix, entropy_floor, bODD, bEVEN, ODDS, EVENS, L, D, VOCAB

device = "cuda"
POINTS = ["emb", "mid0", "post0", "mid1", "post1", "final"]
H = 4
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def load():
    ck = torch.load("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_parity/parity.pt",
                    map_location=device, weights_only=False)
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def resid_labels(model, point, bs=512, tmin=20):
    x, y, isODD = sample_batch_mix(bs, device)
    R = cache_resids(model, x)[point][:, tmin:, :].reshape(-1, 64).numpy()
    lab = isODD[:, None].expand(-1, L - tmin).reshape(-1).cpu().numpy()
    return R, lab


def probe_acc(model, point):
    Xtr, ytr = resid_labels(model, point); Xte, yte = resid_labels(model, point)
    return LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)


@torch.no_grad()
def tf_ce(model, nb=8, bs=256):
    return sum(F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
               for x, y, _ in (sample_batch_mix(bs, device) for _ in range(nb))) / nb


def ov_write(model, b, h):
    blk = model.blocks[b]; mha = blk.mha; dk = mha.lin_v.out_features // H
    with torch.no_grad():
        e = blk.norm1(model.emb.weight[:10])
        v = mha.lin_v(e).view(10, H, dk)
        vm = torch.zeros_like(v); vm[:, h] = v[:, h]
        w = vm.reshape(10, H * dk) @ mha.out.weight.T
    return w


def main():
    model = load(); floor = entropy_floor(torch.tensor(bODD, device=device), torch.tensor(bEVEN, device=device))
    base_ce = tf_ce(model); base_acc = probe_acc(model, "final")
    print(f"baseline: CE={base_ce:.4f} gap={base_ce-floor:.4f}  regime-decodability(final)={100*base_acc:.1f}%")

    # Q1: localization
    loc = {p: probe_acc(model, p) for p in POINTS}
    print("regime decodability by residual point:", {p: round(100 * v, 1) for p, v in loc.items()})

    # Q2: per-head ablation
    heads = [(b, h) for b in (0, 1) for h in range(H)]
    dCE, accAbl = [], []
    for b, h in heads:
        model.blocks[b].mha.ablate_heads = [h]
        dCE.append(tf_ce(model) - base_ce); accAbl.append(probe_acc(model, "final"))
        model.blocks[b].mha.ablate_heads = []
    labels = [f"b{b}h{h}" for b, h in heads]
    top = int(np.argmax(dCE))
    print("per-head ablation ΔCE:", {labels[i]: round(dCE[i], 3) for i in range(len(heads))})
    print(f"most important head: {labels[top]} (ΔCE={dCE[top]:.3f}, regime-decodability when ablated={100*accAbl[top]:.1f}%)")

    # Q3: OV write of the top head projected on the parity axis (final-layer regime probe direction)
    Xtr, ytr = resid_labels(model, "final")
    wdir = LogisticRegression(max_iter=1000).fit(Xtr, ytr).coef_[0]
    wdir = wdir / np.linalg.norm(wdir)
    bT, hT = heads[top]
    proj = (ov_write(model, bT, hT).cpu().numpy() @ wdir)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(POINTS, [100 * loc[p] for p in POINTS], "o-", lw=2, color="#2f8f4f")
    ax[0].axhline(50, ls=":", c="gray"); ax[0].set(ylabel="regime decodability (%)",
        title="Q1: where parity is computed", ylim=(45, 102)); ax[0].tick_params(axis="x", rotation=30)

    x = np.arange(len(heads)); col = ["#3f7fb0"] * 4 + ["#c0464b"] * 4
    ax[1].bar(x, dCE, color=col); ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=45)
    ax[1].set(ylabel="ΔCE when head ablated (nats)", title="Q2: which heads (blue=block0, red=block1)")

    pc = ["#c0464b" if d in ODDS else "#3f7fb0" for d in range(10)]
    ax[2].bar(range(10), proj, color=pc)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="digit", ylabel="OV write · parity axis", xticks=range(10),
              title=f"Q3: head {labels[top]} OV write (red=odd, blue=even)")
    fig.suptitle("Parity circuit: localization → responsible head → OV writes parity")
    fig.tight_layout(); fig.savefig(OUT / "parity_circuit.png", dpi=110); plt.close(fig)
    print("saved parity_circuit.png")


if __name__ == "__main__":
    main()
