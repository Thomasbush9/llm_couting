"""Rarity sweep at L=100: force digit 6 to appear k times, k in {1,2,5,10,20}.

The other 9 digits fill the remaining L-k positions as evenly as possible, so every
sequence is a fixed-multiset permutation with an imbalanced budget. k=10 is the
usual balanced task. Same architecture/optimizer as everywhere else; only the data
budget changes. For each k we train 10k steps and measure the teacher-forced gap to
the (budget-specific) Bayes floor plus free-running decoding -- in particular whether
the model emits digit 6 exactly k times.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_counting.model.model import Transformer

device = "cuda"; D = 10; VOCAB = 11; BOS = 10; L = 100; RARE = 6
KS = [1, 2, 5, 10, 20]
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_digit6"); CK.mkdir(parents=True, exist_ok=True)


def make_budget(k):
    rem = L - k; base = rem // 9; others = [base] * 9
    for i in range(rem - base * 9):
        others[i] += 1
    b, j = [], 0
    for d in range(D):
        if d == RARE: b.append(k)
        else: b.append(others[j]); j += 1
    assert sum(b) == L, (b, sum(b))
    return b


def sample_batch_budget(bs, budget, device):
    base = torch.cat([torch.full((c,), d, dtype=torch.long) for d, c in enumerate(budget)])
    seqs = base[torch.rand(bs, L).argsort(-1)]
    seqs = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), seqs], 1)
    return seqs[:, :-1].to(device), seqs[:, 1:].to(device)


def true_conditional_budget(targets, bt):
    oh = F.one_hot(targets, num_classes=D).float()
    cumsum = torch.cumsum(oh, 1) - oh
    rem = (bt.view(1, 1, D) - cumsum).clamp(min=0)
    return rem / rem.sum(-1, keepdim=True)


@torch.no_grad()
def entropy_floor_budget(bt, nb=100, bs=256):
    tot = 0.0
    for _ in range(nb):
        _, y = sample_batch_budget(bs, bt.tolist(), device)
        tot += torch.special.entr(true_conditional_budget(y, bt)).sum(-1).mean().item()
    return tot / nb


@torch.no_grad()
def val_ce(model, budget, bs=256, nb=10):
    tot = 0.0
    for _ in range(nb):
        x, y = sample_batch_budget(bs, budget, device)
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    return tot / nb


@torch.no_grad()
def decode_eval(model, budget, mode, bs=256, T=1.0):
    bt = torch.tensor(budget, device=device)
    seq = torch.full((bs, 1), BOS, dtype=torch.long, device=device)
    for _ in range(L):
        lg = model(seq)[:, -1, :].clone(); lg[:, BOS] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    gen = seq[:, 1:]
    cnt = torch.zeros(bs, D, dtype=torch.long, device=device).scatter_add_(1, gen, torch.ones_like(gen))
    valid = (cnt == bt).all(1).float().mean().item()
    d6 = cnt[:, RARE].float()
    return valid, d6.mean().item(), d6.std().item()


def train_one(budget, steps=10000, bs=256):
    bt = torch.tensor(budget, device=device)
    model = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                        causal=True, max_len=1024, num_blocks=2).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    floor = entropy_floor_budget(bt)
    hist = {"step": [], "val_ce": [], "gap": []}
    best, best_state = float("inf"), None
    for step in range(steps):
        model.train()
        x, y = sample_batch_budget(bs, budget, device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward(); opt.step()
        if step % 200 == 0:
            model.eval(); vce = val_ce(model, budget)
            hist["step"].append(step); hist["val_ce"].append(vce); hist["gap"].append(vce - floor)
            if vce < best:
                best = vce; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, floor, best, hist


def main():
    results = {}
    for k in KS:
        t0 = time.time(); budget = make_budget(k)
        print(f"\n===== digit6={k}  budget={budget} =====", flush=True)
        model, floor, best, hist = train_one(budget)
        np.savez(CK / f"k{k}.npz", step=np.array(hist["step"]), val_ce=np.array(hist["val_ce"]),
                 gap=np.array(hist["gap"]), floor=floor, k=k, budget=np.array(budget))
        torch.save({"model": model.state_dict(), "budget": budget, "k": k}, CK / f"k{k}.pt")
        ce = val_ce(model, budget)
        gv, gd6, gd6s = decode_eval(model, budget, "greedy")
        sv, sd6, sd6s = decode_eval(model, budget, "sample")
        results[k] = dict(k=k, budget=budget, floor=floor, best_val=best, tf_ce=ce, gap=ce - floor,
                          greedy_valid=gv, sample_valid=sv,
                          greedy_d6=gd6, sample_d6=sd6, sample_d6_std=sd6s, minutes=(time.time() - t0) / 60)
        print(f"k={k}: gap={ce-floor:.4f} greedy_valid={gv:.3f} sample_valid={sv:.3f} "
              f"d6 greedy={gd6:.2f} sample={sd6:.2f}±{sd6s:.2f} (target {k})  ({results[k]['minutes']:.1f}m)", flush=True)
        json.dump({str(kk): v for kk, v in results.items()}, open(CK / "results.json", "w"), indent=2)
        del model; torch.cuda.empty_cache()

    # ---- plots ----
    ks = sorted(results); cmap = plt.cm.plasma(np.linspace(0, 0.85, len(ks)))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c, k in zip(cmap, ks):
        h = dict(np.load(CK / f"k{k}.npz")); g = h["gap"].astype(float); g[g <= 0] = np.nan
        ax[0].plot(h["step"], g, color=c, lw=1.8, label=f"6 appears {k}× (floor {float(h['floor']):.3f})")
    ax[0].set(xlabel="step", ylabel="val CE − Bayes floor (nats)", yscale="log",
              title="convergence vs digit-6 rarity (L=100)"); ax[0].legend(fontsize=8)
    ax[1].plot(ks, [100 * results[k]["greedy_valid"] for k in ks], "o-", lw=2, label="greedy", color="#3f7fb0")
    ax[1].plot(ks, [100 * results[k]["sample_valid"] for k in ks], "s-", lw=2, label="ancestral T=1", color="#2f8f4f")
    ax[1].axvline(10, ls=":", c="gray"); ax[1].text(10.3, 5, "balanced", color="gray", fontsize=8)
    ax[1].set(xlabel="times digit 6 appears (k)", ylabel="valid sequences (%)",
              title="decoding validity vs digit-6 rarity", ylim=(-3, 103)); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT / "sweep_digit6.png", dpi=150); plt.close(fig)

    # digit-6 emission accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, ks, "k--", lw=1, label="target (perfect)")
    ax.plot(ks, [results[k]["greedy_d6"] for k in ks], "o-", color="#3f7fb0", label="greedy")
    ax.errorbar(ks, [results[k]["sample_d6"] for k in ks], yerr=[results[k]["sample_d6_std"] for k in ks],
                fmt="s-", color="#2f8f4f", capsize=3, label="ancestral T=1")
    ax.set(xlabel="target digit-6 count (k)", ylabel="emitted digit-6 count",
           title="does the model emit digit 6 the right number of times?")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "sweep_digit6_emission.png", dpi=150); plt.close(fig)
    print("\nsaved sweep_digit6.png, sweep_digit6_emission.png\nDIGIT6 SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
