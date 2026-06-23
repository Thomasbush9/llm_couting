"""Length x structure: does latent parity-regime inference survive at larger L, where
free-running sampling validity collapses?  Trains the parity mixture at L=300 (same 4:1
odd/even ratio as the L=100 model) and compares regime-inference speed and decoding
validity against the saved L=100 parity model.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from scripts.probing_snippets import cache_resids

device = "cuda"; D = 10; VOCAB = 11; BOS = 10
ODDS = [1, 3, 5, 7, 9]; EVENS = [0, 2, 4, 6, 8]
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_parity")


def budgets(Lval):
    even_c = Lval // 25; odd_c = Lval // 5 - even_c       # 4:1 ratio, sums to L
    bO = [odd_c if d in ODDS else even_c for d in range(D)]
    bE = [odd_c if d in EVENS else even_c for d in range(D)]
    assert sum(bO) == Lval == sum(bE)
    return bO, bE


def multiset(b): return torch.cat([torch.full((c,), d, dtype=torch.long) for d, c in enumerate(b)])


def sample(bs, Lval, bO, bE, pODD=0.5):
    isO = (torch.rand(bs) < pODD)
    bases = torch.where(isO[:, None], multiset(bO)[None, :], multiset(bE)[None, :])
    seqs = torch.gather(bases, 1, torch.rand(bs, Lval).argsort(-1))
    seqs = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), seqs], 1)
    return seqs[:, :-1].to(device), seqs[:, 1:].to(device), isO


def true_cond(targets, t1, t2):
    oh = F.one_hot(targets, D).float(); c = torch.cumsum(oh, 1) - oh
    r1 = t1.view(1, 1, D) - c; r2 = t2.view(1, 1, D) - c
    def lw(rem, t):
        v = (rem >= 0).all(-1)
        ll = (torch.lgamma(t.view(1, 1, D) + 1) - torch.lgamma(rem.clamp(min=0) + 1)).sum(-1) + np.log(0.5)
        return torch.where(v, ll, torch.full_like(ll, -1e30))
    w = torch.softmax(torch.stack([lw(r1, t1), lw(r2, t2)], -1), -1)
    mix = w[..., 0:1] * r1.clamp(min=0) + w[..., 1:2] * r2.clamp(min=0)
    return mix / mix.sum(-1, keepdim=True)


@torch.no_grad()
def floor(Lval, bO, bE, nb=60, bs=256):
    t1, t2 = torch.tensor(bO, device=device), torch.tensor(bE, device=device)
    return sum(torch.special.entr(true_cond(sample(bs, Lval, bO, bE)[1], t1, t2)).sum(-1).mean().item()
               for _ in range(nb)) / nb


@torch.no_grad()
def val_ce(model, Lval, bO, bE, nb=10, bs=256):
    tot = 0.0
    for _ in range(nb):
        x, y, _ = sample(bs, Lval, bO, bE)
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    return tot / nb


@torch.no_grad()
def decode(model, Lval, bO, bE, mode, bs=256, T=1.0):
    t1, t2 = torch.tensor(bO, device=device), torch.tensor(bE, device=device)
    seq = torch.full((bs, 1), BOS, dtype=torch.long, device=device)
    for _ in range(Lval):
        lg = model(seq)[:, -1, :].clone(); lg[:, BOS] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    cnt = torch.zeros(bs, D, dtype=torch.long, device=device).scatter_add_(1, seq[:, 1:], torch.ones_like(seq[:, 1:]))
    return ((cnt == t1).all(1) | (cnt == t2).all(1)).float().mean().item()


def regime_by_pos(model, Lval, bO, bE):
    x, y, isO = sample(512, Lval, bO, bE); R = cache_resids(model, x)["mid1"]
    tmin = 20
    clf = LogisticRegression(max_iter=1000).fit(R[:, tmin:, :].reshape(-1, 64).numpy(),
                                                isO[:, None].expand(-1, Lval - tmin).reshape(-1).cpu().numpy())
    x2, y2, iso2 = sample(512, Lval, bO, bE); R2 = cache_resids(model, x2)["mid1"]
    pred = clf.predict(R2.reshape(-1, 64).numpy()).reshape(512, Lval)
    return (pred == iso2[:, None].cpu().numpy()).mean(0)


def train(Lval, bO, bE, steps=10000, bs=256):
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    best, bs_state = float("inf"), None
    for s in range(steps):
        m.train(); x, y, _ = sample(bs, Lval, bO, bE)
        opt.zero_grad(); F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1)).backward(); opt.step()
        if s % 200 == 0:
            m.eval(); v = val_ce(m, Lval, bO, bE)
            if v < best: best = v; bs_state = {k: val.detach().cpu().clone() for k, val in m.state_dict().items()}
    m.load_state_dict(bs_state); m.eval(); return m, best


def load_L100():
    ck = torch.load(CK / "parity.pt", map_location=device, weights_only=False)
    m = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                    causal=True, max_len=1024, num_blocks=2).to(device)
    m.load_state_dict(ck["model"]); m.eval(); return m


def main():
    t0 = time.time(); Lbig = 300
    bO, bE = budgets(Lbig)
    print(f"L={Lbig} odd-heavy budget={bO}", flush=True)
    fl = floor(Lbig, bO, bE); model, best = train(Lbig, bO, bE)
    torch.save({"model": model.state_dict(), "bODD": bO, "bEVEN": bE, "L": Lbig}, CK / "parity_L300.pt")
    res = {"L": Lbig, "floor": fl, "gap": best - fl,
           "greedy_valid": decode(model, Lbig, bO, bE, "greedy"),
           "sample_valid": decode(model, Lbig, bO, bE, "sample"), "minutes": (time.time() - t0) / 60}
    json.dump(res, open(CK / "results_L300.json", "w"), indent=2)
    print(f"L=300: gap={res['gap']:.4f} greedy={res['greedy_valid']:.3f} sample={res['sample_valid']:.3f}", flush=True)

    # comparison vs L=100
    m100 = load_L100(); bO100, bE100 = budgets(100)
    acc100 = regime_by_pos(m100, 100, bO100, bE100)
    acc300 = regime_by_pos(model, Lbig, bO, bE)
    v100 = decode(m100, 100, bO100, bE100, "sample")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(range(100), 100 * acc100, lw=2, color="#3f7fb0", label="L=100")
    ax[0].plot(range(Lbig), 100 * acc300, lw=2, color="#c0464b", label="L=300")
    ax[0].axhline(50, ls=":", c="gray"); ax[0].set(xlabel="position t", ylabel="regime decodability (%)",
        title="A) parity inference vs position (does length help?)", ylim=(45, 102), xlim=(0, 60)); ax[0].legend()
    ax[1].bar([0, 1], [100 * v100, 100 * res["sample_valid"]], color=["#3f7fb0", "#c0464b"])
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["L=100", "L=300"])
    ax[1].set(ylabel="ancestral-T=1 valid sequences (%)", title="B) decoding validity vs length", ylim=(0, 100))
    for i, v in enumerate([100 * v100, 100 * res["sample_valid"]]):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.suptitle("Parity mixture: length × structure")
    fig.tight_layout(); fig.savefig(OUT / "parity_length.png", dpi=120); plt.close(fig)
    print(f"regime reaches 95% by t={int(np.argmax(acc300>0.95))} (L300), t={int(np.argmax(acc100>0.95))} (L100)", flush=True)
    print(f"sample validity: L100={v100:.3f} L300={res['sample_valid']:.3f}\nsaved parity_length.png\nPARITY-L DONE", flush=True)


if __name__ == "__main__":
    main()
