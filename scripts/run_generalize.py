"""Generalization to held-out budgets.

One model, L=100. Each sequence draws digit-6's budget k uniformly from a TRAIN set
(the other 9 digits fill 100-k); the model must INFER k from the prefix. We hold out
two interior k values and test extrapolation beyond the trained range. Test: give a
real budget-k prefix, let the model greedily complete, and check it emits exactly k 6's.
Generalize => held-out k land on the diagonal; memorize => they don't.
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer

device = "cuda"; D = 10; VOCAB = 11; BOS = 10; L = 100; RARE = 6
TRAIN_K = [k for k in range(1, 21) if k not in (8, 13)]   # 1..20 except 8,13
HELDOUT_INTERIOR = [8, 13]
HELDOUT_EXTRAP = [23, 25]
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_generalize"); CK.mkdir(parents=True, exist_ok=True)
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


def make_budget(k):
    rem = L - k; base = rem // 9; o = [base] * 9
    for i in range(rem - base * 9): o[i] += 1
    b, j = [], 0
    for d in range(D):
        if d == RARE: b.append(k)
        else: b.append(o[j]); j += 1
    return b


def multiset(b): return torch.cat([torch.full((c,), d, dtype=torch.long) for d, c in enumerate(b)])


def sample_train(bs):
    ks = [TRAIN_K[i] for i in torch.randint(0, len(TRAIN_K), (bs,)).tolist()]
    bases = torch.stack([multiset(make_budget(k)) for k in ks])           # (bs,L)
    seqs = torch.gather(bases, 1, torch.rand(bs, L).argsort(-1))
    seqs = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), seqs], 1)
    return seqs[:, :-1].to(device), seqs[:, 1:].to(device)


def train(steps=12000, bs=256):
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    for s in range(steps):
        m.train(); x, y = sample_train(bs)
        opt.zero_grad(); F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1)).backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def conditioned_complete(model, k, prefix_len=60, bs=256):
    """Real budget-k prefix, then greedy completion; count total 6's emitted."""
    base = multiset(make_budget(k))
    full = base[torch.rand(bs, L).argsort(-1)]                            # (bs,L)
    seq = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), full], 1)[:, :prefix_len + 1].to(device)
    for _ in range(L - prefix_len):
        lg = model(seq)[:, -1, :].clone(); lg[:, BOS] = float("-inf")
        seq = torch.cat([seq, lg.argmax(-1)[:, None]], 1)
    c6 = (seq[:, 1:] == RARE).sum(1).float()
    return c6.mean().item(), c6.std().item(), (c6 == k).float().mean().item()


@torch.no_grad()
def tf_track(model, k, bs=256):
    """Teacher-forced P(next=6) vs the known-k Bayes conditional."""
    base = multiset(make_budget(k))
    full = base[torch.rand(bs, L).argsort(-1)]
    x = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), full], 1)[:, :-1].to(device)
    y = full.to(device)
    lg = model(x); lg[..., BOS] = float("-inf")
    pm = torch.softmax(lg, -1)[..., RARE]                                 # (bs,L)
    oh6 = (y == RARE).float(); c6 = torch.cumsum(oh6, 1) - oh6
    R = (L - torch.arange(L, device=device)).float()
    pb = ((k - c6).clamp(min=0) / R)
    return pm.mean(0).cpu().numpy(), pb.mean(0).cpu().numpy()


def main():
    t0 = time.time()
    print(f"TRAIN_K={TRAIN_K}\nheld-out interior={HELDOUT_INTERIOR} extrap={HELDOUT_EXTRAP}", flush=True)
    model = train()
    torch.save({"model": model.state_dict(), "train_k": TRAIN_K}, CK / "gen.pt")

    ks = list(range(1, 27))
    means, stds, exact = {}, {}, {}
    for k in ks:
        m_, s_, e_ = conditioned_complete(model, k)
        means[k], stds[k], exact[k] = m_, s_, e_
    json.dump({"means": means, "exact": exact, "minutes": (time.time() - t0) / 60},
              open(CK / "results.json", "w"), indent=2)
    for tag, group in [("train", TRAIN_K), ("interior", HELDOUT_INTERIOR), ("extrap", HELDOUT_EXTRAP)]:
        print(tag, {k: (round(means[k], 1), round(exact[k], 2)) for k in group}, flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    kk = np.array(ks)
    ax[0].plot(kk, kk, "k--", lw=1, label="perfect (emit = budget)")
    def scat(group, **kw): ax[0].errorbar(group, [means[k] for k in group],
                                          yerr=[stds[k] for k in group], fmt="o", capsize=2, **kw)
    scat(TRAIN_K, color="#3f7fb0", label="train k")
    scat(HELDOUT_INTERIOR, color="#c0464b", ms=9, mec="k", label="held-out interior (8,13)")
    scat(HELDOUT_EXTRAP, color="#e09020", ms=9, mec="k", label="extrapolation (23,25)")
    ax[0].axvspan(0.5, 20.5, color="green", alpha=0.05)
    ax[0].set(xlabel="true digit-6 budget k", ylabel="emitted digit-6 count (greedy completion)",
              title="A) conditioned completion: does it hit held-out budgets?"); ax[0].legend(fontsize=8)

    for k, c in [(5, "#3f7fb0"), (8, "#c0464b"), (13, "#9467bd")]:
        pm, pb = tf_track(model, k)
        lab = "held-out" if k in HELDOUT_INTERIOR else "train"
        ax[1].plot(pm, color=c, lw=2, label=f"k={k} model ({lab})")
        ax[1].plot(pb, color=c, lw=1, ls="--")
    ax[1].set(xlabel="position t", ylabel="P(next = 6)",
              title="B) P(next=6) vs Bayes (dashed) — held-out k=8 tracks too"); ax[1].legend(fontsize=8)
    fig.suptitle("Generalization to held-out digit-6 budgets")
    fig.tight_layout(); fig.savefig(OUT / "generalization.png", dpi=120); plt.close(fig)
    print("saved generalization.png\nGEN DONE", flush=True)


if __name__ == "__main__":
    main()
