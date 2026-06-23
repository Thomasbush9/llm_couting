"""Parity-mixture task at L=100: odd digits dominant or even digits dominant.

50/50 mixture of two regimes (model never sees the label, must infer it):
  ODD-heavy : odds {1,3,5,7,9}=16 each, evens {0,2,4,6,8}=4 each
  EVEN-heavy: evens=16, odds=4
Then we look at HOW the model represents this: (1) the parity-regime is linearly
decodable from the residual and how fast it's inferred over position; (2) the
per-digit count read-out directions cluster by parity.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.linear_model import Ridge, LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from scripts.probing_snippets import cache_resids

device = "cuda"; D = 10; VOCAB = 11; BOS = 10; L = 100
ODDS = [1, 3, 5, 7, 9]; EVENS = [0, 2, 4, 6, 8]
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_parity"); CK.mkdir(parents=True, exist_ok=True)

bODD = [16 if d in ODDS else 4 for d in range(D)]      # odd-heavy regime
bEVEN = [16 if d in EVENS else 4 for d in range(D)]    # even-heavy regime
assert sum(bODD) == L == sum(bEVEN)


def multiset(b): return torch.cat([torch.full((c,), d, dtype=torch.long) for d, c in enumerate(b)])


def sample_batch_mix(bs, device, pODD=0.5):
    isODD = (torch.rand(bs) < pODD)
    bases = torch.where(isODD[:, None], multiset(bODD)[None, :], multiset(bEVEN)[None, :])
    seqs = torch.gather(bases, 1, torch.rand(bs, L).argsort(-1))
    seqs = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), seqs], 1)
    return seqs[:, :-1].to(device), seqs[:, 1:].to(device), isODD


def true_conditional_mix(targets, t1, t2):
    oh = F.one_hot(targets, num_classes=D).float()
    c = torch.cumsum(oh, 1) - oh
    r1 = t1.view(1, 1, D) - c; r2 = t2.view(1, 1, D) - c
    def logw(rem, t):
        valid = (rem >= 0).all(-1)
        ll = (torch.lgamma(t.view(1, 1, D) + 1) - torch.lgamma(rem.clamp(min=0) + 1)).sum(-1) + np.log(0.5)
        return torch.where(valid, ll, torch.full_like(ll, -1e30))
    w = torch.softmax(torch.stack([logw(r1, t1), logw(r2, t2)], -1), -1)
    mixed = w[..., 0:1] * r1.clamp(min=0) + w[..., 1:2] * r2.clamp(min=0)
    return mixed / mixed.sum(-1, keepdim=True)


@torch.no_grad()
def entropy_floor(t1, t2, nb=100, bs=256):
    tot = 0.0
    for _ in range(nb):
        _, y, _ = sample_batch_mix(bs, device)
        tot += torch.special.entr(true_conditional_mix(y, t1, t2)).sum(-1).mean().item()
    return tot / nb


@torch.no_grad()
def val_ce(model, nb=10, bs=256):
    tot = 0.0
    for _ in range(nb):
        x, y, _ = sample_batch_mix(bs, device)        # input and target from the SAME batch
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    return tot / nb


@torch.no_grad()
def decode_eval(model, mode, bs=512, T=1.0):
    t1 = torch.tensor(bODD, device=device); t2 = torch.tensor(bEVEN, device=device)
    seq = torch.full((bs, 1), BOS, dtype=torch.long, device=device)
    for _ in range(L):
        lg = model(seq)[:, -1, :].clone(); lg[:, BOS] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    cnt = torch.zeros(bs, D, dtype=torch.long, device=device).scatter_add_(1, seq[:, 1:], torch.ones_like(seq[:, 1:]))
    mO = (cnt == t1).all(1); mE = (cnt == t2).all(1)
    return dict(valid=(mO | mE).float().mean().item(), fracODD=mO.float().mean().item(), fracEVEN=mE.float().mean().item())


def train(steps=10000, bs=256):
    t1 = torch.tensor(bODD, device=device); t2 = torch.tensor(bEVEN, device=device)
    model = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                        causal=True, max_len=1024, num_blocks=2).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    floor = entropy_floor(t1, t2)
    best, best_state = float("inf"), None
    for step in range(steps):
        model.train()
        x, y, _ = sample_batch_mix(bs, device)
        opt.zero_grad(); F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).backward(); opt.step()
        if step % 200 == 0:
            model.eval(); v = val_ce(model)
            if v < best: best = v; best_state = {k: val.detach().cpu().clone() for k, val in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, floor, best


def counts(targets):
    oh = F.one_hot(targets, num_classes=D).float()
    return (torch.cumsum(oh, 1) - oh)


def main():
    t0 = time.time()
    print(f"ODD-heavy budget ={bODD}\nEVEN-heavy budget={bEVEN}", flush=True)
    model, floor, best = train()
    torch.save({"model": model.state_dict(), "bODD": bODD, "bEVEN": bEVEN}, CK / "parity.pt")
    t1 = torch.tensor(bODD, device=device); t2 = torch.tensor(bEVEN, device=device)
    gd = decode_eval(model, "greedy"); sd = decode_eval(model, "sample")
    res = dict(floor=floor, gap=best - floor, greedy=gd, sample=sd, minutes=(time.time() - t0) / 60)
    json.dump(res, open(CK / "results.json", "w"), indent=2)
    print(f"floor={floor:.4f} gap={best-floor:.4f}", flush=True)
    print(f"greedy valid={gd['valid']:.3f} (ODD={gd['fracODD']:.2f} EVEN={gd['fracEVEN']:.2f})", flush=True)
    print(f"sample valid={sd['valid']:.3f} (ODD={sd['fracODD']:.2f} EVEN={sd['fracEVEN']:.2f})", flush=True)

    # ---- (1) parity regime inference: P(next in odd) vs P(next in even) ----
    torch.manual_seed(5)
    def ex(pODD):
        x, y, _ = sample_batch_mix(1, device, pODD=pODD)
        with torch.no_grad():
            lg = model(x)[0]
        m = lg.clone(); m[:, BOS] = float("-inf"); Pm = torch.softmax(m, -1)[:, :D]
        Pt = true_conditional_mix(y, t1, t2)[0]
        po = lambda P: P[:, ODDS].sum(-1).cpu().numpy(); pe = lambda P: P[:, EVENS].sum(-1).cpu().numpy()
        return (po(Pm), pe(Pm), po(Pt), pe(Pt))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for a, (pODD, lab) in zip(ax, [(1.0, "ODD-heavy sequence"), (0.0, "EVEN-heavy sequence")]):
        mo, me, to, te = ex(pODD)
        a.plot(mo, color="#c0464b", lw=2, label="model P(next ∈ odd)")
        a.plot(to, color="#c0464b", lw=1, ls="--", label="Bayes P(next ∈ odd)")
        a.plot(me, color="#3f7fb0", lw=2, label="model P(next ∈ even)")
        a.plot(te, color="#3f7fb0", lw=1, ls="--", label="Bayes P(next ∈ even)")
        a.set(xlabel="position t", title=lab); a.legend(fontsize=8)
    ax[0].set_ylabel("total probability mass")
    fig.suptitle("Parity-regime inference: model infers odd-heavy vs even-heavy from the prefix")
    fig.tight_layout(); fig.savefig(OUT / "parity_regime_inference.png", dpi=150); plt.close(fig)

    # ---- (2) representation: regime decodability + count-direction parity geometry ----
    xtr, ytr, lbltr = sample_batch_mix(512, device)
    xte, yte, lblte = sample_batch_mix(512, device)
    Ptr = cache_resids(model, xtr); Pte = cache_resids(model, xte)
    Xtr = Ptr["mid1"].reshape(-1, 64).numpy(); Xte = Pte["mid1"].reshape(-1, 64).numpy()

    # 2a: linear probe for the regime label (odd-heavy=1), accuracy as a function of position
    ytr_lbl = lbltr[:, None].expand(-1, L).reshape(-1).cpu().numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr_lbl)
    pred = clf.predict(Pte["mid1"].reshape(-1, 64).numpy()).reshape(512, L)
    truth = lblte[:, None].expand(-1, L).cpu().numpy()
    acc_t = (pred == truth).mean(0)

    # 2b: per-digit count read-out directions at mid1, cosine sim, reordered by parity
    Rd = Ridge(1.0).fit(Xtr, counts(ytr).reshape(-1, D).cpu().numpy()).coef_   # (10,64)
    Rn = Rd / (np.linalg.norm(Rd, axis=1, keepdims=True) + 1e-9)
    order = EVENS + ODDS
    C = (Rn @ Rn.T)[np.ix_(order, order)]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    ax[0].plot(range(L), 100 * acc_t, lw=2, color="#2f8f4f")
    ax[0].axhline(50, ls=":", c="gray", label="chance"); ax[0].set(xlabel="position t",
        ylabel="regime decodability (%)", title="linear probe: odd-heavy vs even-heavy from residual", ylim=(45, 102))
    ax[0].legend()
    im = ax[1].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[1].set_xticks(range(10)); ax[1].set_xticklabels(order); ax[1].set_yticks(range(10)); ax[1].set_yticklabels(order)
    ax[1].axhline(4.5, color="k", lw=1); ax[1].axvline(4.5, color="k", lw=1)
    ax[1].set(title="cosine of per-digit count directions\n(reordered: evens | odds)", xlabel="digit", ylabel="digit")
    fig.colorbar(im, ax=ax[1])
    fig.suptitle("How the model represents parity")
    fig.tight_layout(); fig.savefig(OUT / "parity_representation.png", dpi=150); plt.close(fig)
    print(f"regime decodable by t=10: {100*acc_t[10]:.1f}%  by t=30: {100*acc_t[30]:.1f}%  final: {100*acc_t[-1]:.1f}%", flush=True)
    print("saved parity_regime_inference.png, parity_representation.png\nPARITY DONE", flush=True)


if __name__ == "__main__":
    main()
