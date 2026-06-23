"""Mixture/latent-regime task at L=100: digits 6 and 3 are anti-correlated.

Each sequence is drawn from one of two regimes (50/50):
  A: digit6=20, digit3=2     B: digit6=2, digit3=20
the other 8 digits are identical (~10 each). The model never sees the regime label
-- it must INFER it from the prefix (seeing many 6's => regime A => digit 3 is rare).
The Bayes-optimal next-token dist is the posterior-weighted mix of the two budgets.

Trains one model (same arch as everywhere), then: decoding validity (does a generated
sequence match A or B exactly?), and figures showing the model tracking the Bayes
regime inference (P(next=3) vs P(next=6)) plus the digit x position matrices.
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

device = "cuda"; D = 10; VOCAB = 11; BOS = 10; L = 100
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_mix36"); CK.mkdir(parents=True, exist_ok=True)

# ---- the two regimes: 6 and 3 swap between {20, 2}; others identical ----
OTHERS = {0: 10, 1: 10, 2: 10, 4: 10, 5: 10, 7: 10, 8: 9, 9: 9}   # sum 78
bA = [0] * D; bB = [0] * D
for d, v in OTHERS.items(): bA[d] = v; bB[d] = v
bA[6], bA[3] = 20, 2
bB[6], bB[3] = 2, 20
assert sum(bA) == L and sum(bB) == L


def multiset(budget):
    return torch.cat([torch.full((c,), d, dtype=torch.long) for d, c in enumerate(budget)])


def sample_batch_mix(bs, device, pA=0.5):
    isA = (torch.rand(bs) < pA)
    bases = torch.where(isA[:, None], multiset(bA)[None, :], multiset(bB)[None, :])  # (bs,L)
    seqs = torch.gather(bases, 1, torch.rand(bs, L).argsort(-1))
    seqs = torch.cat([torch.full((bs, 1), BOS, dtype=torch.long), seqs], 1)
    return seqs[:, :-1].to(device), seqs[:, 1:].to(device), isA


def true_conditional_mix(targets, tA, tB, logpA=np.log(0.5), logpB=np.log(0.5)):
    """Posterior-weighted Bayes next-token distribution over the two regimes."""
    oh = F.one_hot(targets, num_classes=D).float()
    c = torch.cumsum(oh, 1) - oh                                  # (B,L,D) exclusive counts
    remA = tA.view(1, 1, D) - c; remB = tB.view(1, 1, D) - c
    def logw(rem, t, logp):
        valid = (rem >= 0).all(-1)
        ll = (torch.lgamma(t.view(1, 1, D) + 1) - torch.lgamma(rem.clamp(min=0) + 1)).sum(-1) + logp
        return torch.where(valid, ll, torch.full_like(ll, -1e30))
    w = torch.softmax(torch.stack([logw(remA, tA, logpA), logw(remB, tB, logpB)], -1), -1)  # (B,L,2)
    mixed = w[..., 0:1] * remA.clamp(min=0) + w[..., 1:2] * remB.clamp(min=0)
    return mixed / mixed.sum(-1, keepdim=True), w                # (B,L,D), (B,L,2)


@torch.no_grad()
def entropy_floor_mix(tA, tB, nb=100, bs=256):
    tot = 0.0
    for _ in range(nb):
        _, y, _ = sample_batch_mix(bs, device)
        p, _ = true_conditional_mix(y, tA, tB)
        tot += torch.special.entr(p).sum(-1).mean().item()
    return tot / nb


@torch.no_grad()
def val_ce(model, nb=10, bs=256):
    tot = 0.0
    for _ in range(nb):
        x, y, _ = sample_batch_mix(bs, device)
        tot += F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
    return tot / nb


@torch.no_grad()
def decode_eval(model, mode, bs=512, T=1.0):
    btA = torch.tensor(bA, device=device); btB = torch.tensor(bB, device=device)
    seq = torch.full((bs, 1), BOS, dtype=torch.long, device=device)
    for _ in range(L):
        lg = model(seq)[:, -1, :].clone(); lg[:, BOS] = float("-inf")
        nxt = lg.argmax(-1) if mode == "greedy" else torch.multinomial(torch.softmax(lg / T, -1), 1).squeeze(-1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    cnt = torch.zeros(bs, D, dtype=torch.long, device=device).scatter_add_(1, seq[:, 1:], torch.ones_like(seq[:, 1:]))
    mA = (cnt == btA).all(1); mB = (cnt == btB).all(1)
    return dict(valid=(mA | mB).float().mean().item(), fracA=mA.float().mean().item(),
                fracB=mB.float().mean().item())


def train(steps=10000, bs=256):
    tA = torch.tensor(bA, device=device); tB = torch.tensor(bB, device=device)
    model = Transformer(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                        causal=True, max_len=1024, num_blocks=2).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    floor = entropy_floor_mix(tA, tB)
    hist = {"step": [], "gap": []}; best, best_state = float("inf"), None
    for step in range(steps):
        model.train()
        x, y, _ = sample_batch_mix(bs, device)
        opt.zero_grad(); F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).backward(); opt.step()
        if step % 200 == 0:
            model.eval(); vce = val_ce(model)
            hist["step"].append(step); hist["gap"].append(vce - floor)
            if vce < best: best = vce; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    return model, floor, best, hist


def main():
    t0 = time.time()
    print(f"regime A budget={bA}\nregime B budget={bB}", flush=True)
    model, floor, best, hist = train()
    torch.save({"model": model.state_dict(), "bA": bA, "bB": bB}, CK / "mix36.pt")
    tA = torch.tensor(bA, device=device); tB = torch.tensor(bB, device=device)
    gd = decode_eval(model, "greedy"); sd = decode_eval(model, "sample")
    res = dict(floor=floor, best_val=best, gap=best - floor, greedy=gd, sample=sd, minutes=(time.time() - t0) / 60)
    json.dump(res, open(CK / "results.json", "w"), indent=2)
    print(f"floor={floor:.4f} gap={best-floor:.4f}", flush=True)
    print(f"greedy: valid={gd['valid']:.3f} (A={gd['fracA']:.2f} B={gd['fracB']:.2f})", flush=True)
    print(f"sample: valid={sd['valid']:.3f} (A={sd['fracA']:.2f} B={sd['fracB']:.2f})", flush=True)

    # ---- example sequences from each regime (teacher-forced) ----
    torch.manual_seed(7)
    def example(force_pA):
        x, y, _ = sample_batch_mix(1, device, pA=force_pA)
        with torch.no_grad():
            lg = model(x)[0]
        masked = lg.clone(); masked[:, BOS] = float("-inf")
        Pm = torch.softmax(masked, -1)[:, :D]
        Pt, w = true_conditional_mix(y, tA, tB)
        return y[0].cpu().numpy(), Pm.cpu().numpy(), Pt[0].cpu().numpy(), w[0].cpu().numpy()

    yA, PmA, PtA, wA = example(1.0)   # regime A (6 high, 3 low)
    yB, PmB, PtB, wB = example(0.0)   # regime B (6 low, 3 high)

    # ---- Fig 1: regime inference — P(next=6) and P(next=3) tracking Bayes ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for a, (y, Pm, Pt, lab) in zip(ax, [(yA, PmA, PtA, "regime A (true: 6=20, 3=2)"),
                                        (yB, PmB, PtB, "regime B (true: 6=2, 3=20)")]):
        a.plot(Pm[:, 6], color="#c0464b", lw=2, label="model P(next=6)")
        a.plot(Pt[:, 6], color="#c0464b", lw=1, ls="--", label="Bayes P(next=6)")
        a.plot(Pm[:, 3], color="#3f7fb0", lw=2, label="model P(next=3)")
        a.plot(Pt[:, 3], color="#3f7fb0", lw=1, ls="--", label="Bayes P(next=3)")
        a.set(xlabel="position t", title=lab); a.legend(fontsize=8)
    ax[0].set_ylabel("P(next = digit)")
    fig.suptitle("Latent-regime inference: model tracks Bayes as it figures out the regime from the prefix")
    fig.tight_layout(); fig.savefig(OUT / "mix36_regime_inference.png", dpi=150); plt.close(fig)

    # ---- Fig 2: digit x position matrices for A and B ----
    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    vmax = max(PmA.max(), PmB.max())
    for a, (Pm, lab) in zip(ax, [(PmA, "regime A sample (6 high, 3 low)"), (PmB, "regime B sample (6 low, 3 high)")]):
        im = a.imshow(Pm.T, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=vmax, interpolation="nearest")
        fig.colorbar(im, ax=a, pad=0.01); a.set_yticks(range(10)); a.set_ylabel("digit"); a.set_title(lab)
        a.axhline(6, color="red", lw=0.8, ls=":"); a.axhline(3, color="cyan", lw=0.8, ls=":")
    ax[-1].set_xlabel("position t")
    fig.suptitle("digit × position model P(next) — rows 6 (red) and 3 (cyan) swap roles by regime")
    fig.tight_layout(); fig.savefig(OUT / "mix36_prob_matrix.png", dpi=150); plt.close(fig)
    print("saved mix36_regime_inference.png, mix36_prob_matrix.png\nMIX36 DONE", flush=True)


if __name__ == "__main__":
    main()
