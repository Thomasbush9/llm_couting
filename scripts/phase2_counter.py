"""Phase 2 - reverse-engineer the block-0 attention counter (weight + causal level).

Phase 1 said: counts + position appear right after block-0 attention, which looks
like a near-uniform mean-pooler. Here we prove it causally and read the weights:

  (1) OV write geometry: each block-0 head's per-digit write vectors v_{h,d}. Are the
      10 digit tallies linearly separable (distinct directions)?  -> counting is possible.

  (2) UNIFORM-ATTENTION SWAP (causal): force block-0 attention to exact uniform
      1/(t+1) over the causal prefix (ignore the learned Q/K entirely). If the count
      code at mid0 and the Bayes match at the output survive, block-0 IS a mean-pooler.

  (3) BOS-VALUE ABLATION (causal): zero the BOS token's value contribution in block-0.
      If position decodability and the Bayes match collapse, the BOS channel is the
      model's length/position signal (the 1/(t+1) curve from Phase 1).

We report, for clean / uniform / bos-ablated:  count R2 and pos R2 at mid0 (using a
probe fit on the CLEAN stream), and KL(true||model) at the output.
Outputs: phase2_counter.png  (+ printed table).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.train.trainer import sample_batch, true_conditional
from scripts.probing_snippets import load_model, cache_resids, counts_from_targets

device = "cuda"; D = 10
CK = "/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50/best.pt"
OUT = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")


# ----- block-0 attention interventions, implemented as a forward hook that REPLACES
#       the mha output (recomputed from the same input + weights). Registered before
#       cache_resids' hooks so the whole downstream sees the modified output. -----
def make_hook(mha, mode):
    H, d_k = mha.num_heads, mha.d_k

    def hook(mod, inp, out):
        x = inp[0]                                          # (B,L,in_dim), already norm1'd
        B, L, _ = x.shape
        V = rearrange(mha.lin_v(x), 'b s (n h) -> b n s h', n=H, h=d_k)
        if mode == "bos_ablate":                           # real weights, but BOS carries no value
            score = mha.attn.to(x.device)                  # (B,H,L,L) cached from the real pass
            Vz = V.clone(); Vz[:, :, 0, :] = 0.0
            o = torch.einsum('bnqk,bnkh->bnqh', score, Vz)
        else:                                              # uniform 1/(t+1) over causal prefix
            w = torch.tril(torch.ones(L, L, device=x.device))
            w = w / w.sum(-1, keepdim=True)
            o = torch.einsum('qk,bnkh->bnqh', w, V)
        o = rearrange(o, 'b n s h -> b s (n h)')
        return mha.out(o)
    return hook


def _mid0(model, mha, mode, x):
    """Cache mid0 (and run the forward) under the given block-0 intervention."""
    if mode == "bos_ablate":
        with torch.no_grad():
            model(x)                                        # populate mha.attn for the hook
    h = None if mode == "clean" else mha.register_forward_hook(make_hook(mha, mode))
    pts = cache_resids(model, x)
    if h: h.remove()
    return pts["mid0"]


def _cft(y):
    """count (B*,D), freq (B*,D), pos (B*,1) targets for positions t>=1."""
    c = counts_from_targets(y).cpu(); L = c.shape[1]
    t = torch.arange(L).float().view(1, L, 1)
    f = c / t.clamp(min=1)
    return (c[:, 1:].reshape(-1, D).numpy(), f[:, 1:].reshape(-1, D).numpy(),
            t.expand(c.shape[0], L, 1)[:, 1:].reshape(-1, 1).numpy())


def measure(model, mha, mode, probe_c, probe_t, xtr, ytr, xte, yte):
    Mtr = _mid0(model, mha, mode, xtr); Mte = _mid0(model, mha, mode, xte)
    Xtr = Mtr[:, 1:].reshape(-1, Mtr.shape[-1]).numpy()
    Xte = Mte[:, 1:].reshape(-1, Mte.shape[-1]).numpy()
    ctr, ftr, _ = _cft(ytr); cte, fte, t_te = _cft(yte)

    # (a) transfer: clean-fit probe applied to this rep  -> "is the rep UNCHANGED?"
    tr_c = r2_score(cte, probe_c.predict(Xte), multioutput="uniform_average")
    tr_t = r2_score(t_te, probe_t.predict(Xte))
    # (b) refit: fresh probe on this rep -> "does this rep still ENCODE count / freq at all?"
    rf_c = r2_score(cte, Ridge(1.0).fit(Xtr, ctr).predict(Xte), multioutput="uniform_average")
    rf_f = r2_score(fte, Ridge(1.0).fit(Xtr, ftr).predict(Xte), multioutput="uniform_average")

    # KL(true||model) at the output, averaged over positions t>=1
    with torch.no_grad():
        if mode == "bos_ablate":
            model(xte)
        h2 = None if mode == "clean" else mha.register_forward_hook(make_hook(mha, mode))
        lg = model(xte).clone()
        if h2: h2.remove()
    lg[..., -1] = float("-inf")
    pm = torch.softmax(lg, -1)[..., :D].cpu()
    pt = true_conditional(yte.cpu(), VOCAB=D + 1)
    kl = (torch.xlogy(pt, pt) - torch.xlogy(pt, pm.clamp(min=1e-12))).sum(-1)[:, 1:].mean().item()
    return dict(tr_c=tr_c, tr_t=tr_t, rf_c=rf_c, rf_f=rf_f, kl=kl)


def ov_write(model, h, H):
    blk, mha = model.blocks[0], model.blocks[0].mha
    d_k = mha.lin_v.out_features // H
    with torch.no_grad():
        e = blk.norm1(model.emb.weight[:D])
        v = mha.lin_v(e).view(D, H, d_k)
        vm = torch.zeros_like(v); vm[:, h] = v[:, h]
        w = vm.reshape(D, H * d_k) @ mha.out.weight.T
    return (w / w.norm(dim=1, keepdim=True)).cpu()


def main():
    t0 = time.time()
    model, ck = load_model(CK, device)
    L = ck["args"]["seq_length"]; H = ck["model_args"]["num_heads"]
    mha = model.blocks[0].mha
    print(f"checkpoint L={L} H={H}")

    # fit count/position probes on the CLEAN stream
    xtr, ytr = sample_batch(512, L, device=device)
    Ptr = cache_resids(model, xtr)
    Xtr = Ptr["mid0"][:, 1:].reshape(-1, Ptr["mid0"].shape[-1]).numpy()
    ctr = counts_from_targets(ytr).cpu()[:, 1:].reshape(-1, D).numpy()
    ttr = torch.arange(L).float().view(1, -1, 1).expand(512, -1, 1)[:, 1:].reshape(-1, 1).numpy()
    probe_c = Ridge(1.0).fit(Xtr, ctr); probe_t = Ridge(1.0).fit(Xtr, ttr)

    xte, yte = sample_batch(512, L, device=device)
    xtr2, ytr2 = sample_batch(512, L, device=device)
    modes = ["clean", "uniform", "bos_ablate"]
    rows = {}
    for mode in modes:
        r = measure(model, mha, mode, probe_c, probe_t, xtr2, ytr2, xte, yte)
        rows[mode] = r
        print(f"  {mode:<11} transfer[count {r['tr_c']:.3f} pos {r['tr_t']:.3f}]  "
              f"refit[count {r['rf_c']:.3f} freq {r['rf_f']:.3f}]  KL={r['kl']:.4f}")

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # panel A: per-head OV write geometry
    cos = ov_write(model, 0, H) @ ov_write(model, 0, H).T
    im = ax[0].imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set(title="A) block-0 head-0 per-digit write vectors\n(cosine; off-diag ~0 => separable tallies)",
              xlabel="digit", ylabel="digit", xticks=range(D), yticks=range(D))
    fig.colorbar(im, ax=ax[0], shrink=0.8)
    # panel B: transfer vs refit for count, + refit freq
    xpos = np.arange(len(modes))
    ax[1].bar(xpos - 0.27, [rows[m]["tr_c"] for m in modes], 0.25, color="#c0464b", label="count R2 (clean-fit probe)")
    ax[1].bar(xpos + 0.00, [rows[m]["rf_c"] for m in modes], 0.25, color="#e08a8d", label="count R2 (re-fit)")
    ax[1].bar(xpos + 0.27, [rows[m]["rf_f"] for m in modes], 0.25, color="#3f7fb0", label="freq R2 (re-fit)")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(xpos); ax[1].set_xticklabels(modes)
    ax[1].set(ylim=(-0.3, 1.05), ylabel="R2 at mid0",
              title="B) count code: transferable vs re-fittable\n(clean-fit fails but re-fit holds => rep merely rescaled)")
    ax[1].legend(fontsize=7)
    # panel C: KL to Bayes
    ax[2].bar(xpos, [rows[m]["kl"] for m in modes], color=["#3f7fb0", "#9467bd", "#e09020"])
    ax[2].set_xticks(xpos); ax[2].set_xticklabels(modes)
    ax[2].set(title="C) output match to Bayes\n(lower = closer to optimal)", ylabel="KL(true||model)")
    for i, m in enumerate(modes): ax[2].text(i, rows[m]["kl"], f"{rows[m]['kl']:.3f}", ha="center", va="bottom")
    fig.suptitle("Phase 2: block-0 attention as the counter (causal interventions on the L=50 GELU model)")
    fig.tight_layout(); fig.savefig(OUT / "phase2_counter.png", dpi=140); plt.close(fig)
    print(f"saved phase2_counter.png\nPHASE2 DONE ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
