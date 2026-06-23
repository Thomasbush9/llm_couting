"""Phase 1 - representation ID for the counting circuit.

Two questions that frame everything downstream:

  (1) Does the residual stream carry the RAW count c_d, the FREQUENCY f_d = c_d/t,
      or the POSITION t -- and at which residual point does each appear?
      Uniform softmax attention over a length-t prefix returns the MEAN of value
      vectors = the frequency vector, so the prediction is: f_d is linearly present
      right after block-0 attention (mid0); raw c_d needs t as well.

  (2) Is block-0 (and block-1) attention actually a content-independent MEAN-POOLER?
      Tests: attention ~ uniform 1/(t+1) over the causal prefix (entropy ~ log(t+1)),
      weight on BOS ~ 1/(t+1), and low across-batch variability at matched (q,k)
      (positional, not content, driven). A separate head whose BOS weight tracks
      1/(t+1) would be the model's length/position signal.

Outputs: phase1_repr.png, phase1_attn.png  (+ printed R2 table).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.train.trainer import sample_batch
from scripts.probing_snippets import load_model, cache_resids, counts_from_targets

device = "cuda"; D = 10
CK = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50/best.pt"
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
POINTS = ["emb", "mid0", "post0", "mid1", "post1"]


def repr_targets(targets):
    """Exclusive-prefix count c_d, frequency f_d=c_d/t, position t. (B,L,*)"""
    c = counts_from_targets(targets).cpu()                 # (B,L,D)
    L = c.shape[1]
    t = torch.arange(L).float().view(1, L, 1)
    f = c / t.clamp(min=1)                                 # undefined at t=0 (masked out below)
    return c, f, t.expand(c.shape[0], L, 1)


def flat_from1(a):
    """Drop position t=0 (where frequency is undefined) and flatten over (B, pos)."""
    return a[:, 1:].reshape(-1, a.shape[-1]).numpy()


def probe(model, batch, L):
    xtr, ytr = sample_batch(batch, L, device=device)
    xte, yte = sample_batch(batch, L, device=device)
    Ptr, Pte = cache_resids(model, xtr), cache_resids(model, xte)
    ctr, ftr, ttr = repr_targets(ytr); cte, fte, tte = repr_targets(yte)
    tgt_tr = {"count": flat_from1(ctr), "freq": flat_from1(ftr), "pos": flat_from1(ttr)}
    tgt_te = {"count": flat_from1(cte), "freq": flat_from1(fte), "pos": flat_from1(tte)}

    table = {}
    for name in POINTS:
        Xtr = flat_from1(Ptr[name]); Xte = flat_from1(Pte[name])
        row = {}
        for k in ("count", "freq", "pos"):
            p = Ridge(alpha=1.0).fit(Xtr, tgt_tr[k])
            row[k] = r2_score(tgt_te[k], p.predict(Xte), multioutput="uniform_average")
        table[name] = row
        print(f"  {name:<6}  count R2={row['count']:.4f}  freq R2={row['freq']:.4f}  pos R2={row['pos']:.4f}")
    return table


@torch.no_grad()
def attn_diag(model, x, block):
    model(x)
    A = model.blocks[block].mha.attn.cpu()                 # (B,H,L,L)
    B, H, L, _ = A.shape
    q = torch.arange(L).float()
    bos = A[:, :, :, 0].mean(0)                            # (H,L) weight on BOS vs query pos
    Acl = A.clamp(min=1e-12)
    ent = -(Acl * Acl.log()).sum(-1).mean(0)              # (H,L) attention entropy
    causal = torch.tril(torch.ones(L, L)).bool()
    std = A.std(0); mean = A.mean(0).clamp(min=1e-9)
    cv = (std / mean)[:, causal].mean(-1)                  # (H,) across-batch coeff of variation
    return bos.numpy(), ent.numpy(), cv.numpy(), q.numpy()


def main():
    t0 = time.time()
    model, ck = load_model(CK, device)
    L = ck["args"]["seq_length"]; H = ck["model_args"]["num_heads"]
    print(f"checkpoint step={ck['step']}  L={L}  D={D}  H={H}\n[A] representation probe (positions t>=1)")
    table = probe(model, 512, L)

    # ---- Fig A: representation R2 across the stream ----
    fig, ax = plt.subplots(figsize=(8, 5))
    xpos = np.arange(len(POINTS))
    for k, c, off in [("count", "#c0464b", -0.25), ("freq", "#3f7fb0", 0.0), ("pos", "#2f8f4f", 0.25)]:
        ax.bar(xpos + off, [table[n][k] for n in POINTS], 0.25, color=c, label=k)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(POINTS)
    ax.set(ylabel="held-out R^2 (linear probe)", ylim=(-0.05, 1.05),
           title="A) what is linearly present across the residual stream\n(count c_d / frequency c_d/t / position t)")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "phase1_repr.png", dpi=140); plt.close(fig)
    print("saved phase1_repr.png")

    # ---- Fig B: attention shape (is it a mean-pooler?) ----
    print("\n[B] attention diagnostics")
    x, _ = sample_batch(256, L, device=device)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    q = np.arange(L); unif_bos = 1.0 / (q + 1); unif_ent = np.log(q + 1)
    colors = plt.cm.tab10(np.arange(H))
    cv_all = {}
    for blk, ls in [(0, "-"), (1, "--")]:
        bos, ent, cv, _ = attn_diag(model, x, blk)
        cv_all[blk] = cv
        for h in range(H):
            ax[0].plot(q[1:], bos[h, 1:], ls, color=colors[h], lw=1.5,
                       label=f"b{blk} h{h}" if blk == 0 else None)
            ax[1].plot(q[1:], ent[h, 1:], ls, color=colors[h], lw=1.5)
        print(f"  block{blk} across-batch CV per head:", [round(float(c), 3) for c in cv])
    ax[0].plot(q[1:], unif_bos[1:], "k:", lw=2.5, label="uniform 1/(t+1)")
    ax[0].set(xlabel="query position t", ylabel="attention weight on BOS",
              title="i) BOS weight vs uniform (length signal?)"); ax[0].legend(fontsize=7, ncol=2)
    ax[1].plot(q[1:], unif_ent[1:], "k:", lw=2.5, label="uniform log(t+1)")
    ax[1].set(xlabel="query position t", ylabel="attention entropy (nats)",
              title="ii) entropy vs uniform ceiling\n(solid=block0, dashed=block1)"); ax[1].legend(fontsize=8)
    width = 0.35
    hx = np.arange(H)
    ax[2].bar(hx - width / 2, cv_all[0], width, color="#3f7fb0", label="block 0")
    ax[2].bar(hx + width / 2, cv_all[1], width, color="#c0464b", label="block 1")
    ax[2].set(xlabel="head", ylabel="across-batch CV of attention\n(0 = purely positional)",
              title="iii) content dependence per head", xticks=hx); ax[2].legend()
    fig.suptitle("Phase 1B: is block-0/1 attention a content-independent mean-pooler?")
    fig.tight_layout(); fig.savefig(OUT / "phase1_attn.png", dpi=140); plt.close(fig)
    print(f"saved phase1_attn.png\nPHASE1 DONE ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
