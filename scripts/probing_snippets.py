"""Mechanistic probing of the counting transformer.

Refactored from notebook cells into a runnable script. Loads a checkpoint and
produces a sequence of figures into --outdir:

  1. count_probe_split.png   linear count decodability across the residual stream
  2. block0_heads.png        block-0 attention maps (position x position) per head
  3. block0_digit_attn.png   mean attention by (query digit, key digit) per head
  4. head2_OV_geometry.png   cosine geometry of head-2 per-digit write vectors
  5. head2_loop_closure.png  count-readout vs digit-write alignment + decodability
  6. steer_digit.png         steering along a digit's write direction (static)
  7. steer_dynamics.png      the same steering resolved over sequence position

Plus several printed diagnostics (head-ablation R2, OV singular values, ...).

The story it tells: per-digit counts become linearly decodable right after
block-0 attention; head 2 carries a depletion/count signal whose write
directions line up with the count read-out, and injecting along a digit's write
direction steers that digit's next-token probability.

Run:  python scripts/probing_snippets.py --device cuda
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_counting.model.model import Transformer
from llm_counting.train.trainer import sample_batch

D = 10
RESID_POINTS = ["emb", "mid0", "post0", "mid1", "post1", "final"]
HEAD = 2          # the OV-circuit study focuses on block-0 head 2
STEER_DIGIT = 6   # digit whose write direction we steer along


# ----------------------------------------------------------------------------- helpers
def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["model_args"]); cfg.pop("VOCAB", None)
    model = Transformer(**cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def counts_from_targets(targets):
    """Exclusive prefix counts c(t): (B, L, D) -- same alignment as true_conditional."""
    oh = F.one_hot(targets, num_classes=D).float()
    return torch.cumsum(oh, dim=1) - oh


def dev_targets(tg):
    """Counts with the deterministic positional ramp t/D removed -> per-digit deviation."""
    c = counts_from_targets(tg).cpu()                  # (B, L, D)
    L = c.shape[1]
    dev = c - torch.arange(L).view(1, L, 1) / D
    return dev.reshape(-1, D).numpy()


@torch.no_grad()
def cache_resids(model, inputs):
    """Hook residual points + sublayer deltas, return the reconstructed stream."""
    raw = {}

    def save(name):
        def hook(m, i, o):
            raw[name] = o.detach().cpu().float()
        return hook

    handles = [model.emb.register_forward_hook(save("emb"))]
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(save(f"block{i}")))          # resid_post
        handles.append(blk.mha.register_forward_hook(save(f"attn_delta{i}")))
        handles.append(blk.mlp.register_forward_hook(save(f"mlp_delta{i}")))
    handles.append(model.norm.register_forward_hook(save("final")))

    model.eval(); model(inputs)
    for h in handles:
        h.remove()

    pts = {
        "emb":   raw["emb"],                          # resid entering block 0  (control)
        "mid0":  raw["emb"]   + raw["attn_delta0"],   # after block-0 ATTENTION
        "post0": raw["block0"],                       # after block-0 MLP
        "mid1":  raw["block0"] + raw["attn_delta1"],  # after block-1 ATTENTION
        "post1": raw["block1"],                       # after block-1 MLP
        "final": raw["final"],                        # post final norm
    }
    assert torch.allclose(pts["post0"], pts["mid0"] + raw["mlp_delta0"], atol=1e-4)
    return pts


# ----------------------------------------------------------------------------- 1. count probe
def count_probe(model, ck, batch, L, device, out, alpha=1.0):
    xtr, ytr = sample_batch(batch, L, device=device)
    xte, yte = sample_batch(batch, L, device=device)
    pts_tr, pts_te = cache_resids(model, xtr), cache_resids(model, xte)
    ctr = counts_from_targets(ytr).reshape(-1, D).cpu().numpy()
    cte = counts_from_targets(yte).reshape(-1, D).cpu().numpy()

    r2 = {}
    for name in RESID_POINTS:
        Xtr = pts_tr[name].reshape(-1, pts_tr[name].shape[-1]).numpy()
        Xte = pts_te[name].reshape(-1, pts_te[name].shape[-1]).numpy()
        probe = Ridge(alpha=alpha).fit(Xtr, ctr)
        r2[name] = r2_score(cte, probe.predict(Xte))
        print(f"  {name:<6} held-out R2 = {r2[name]:.4f}")

    fig, ax = plt.subplots()
    ax.plot(RESID_POINTS, [r2[n] for n in RESID_POINTS], marker="o")
    ax.axhline(0, ls="--", c="gray")
    ax.set(xlabel="residual point", ylabel="held-out R^2  (count decodability)",
           ylim=(-0.05, 1.05), title=f"linear count probe (step {ck['step']})")
    plt.xticks(rotation=30)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 2. attention maps
@torch.no_grad()
def attn_head_maps(model, xte, out):
    model(xte[:1])
    A = model.blocks[0].mha.attn[0].cpu()[:, 1:, 1:]      # (n_heads, L, L), BOS dropped
    fig, axes = plt.subplots(1, A.shape[0], figsize=(4 * A.shape[0], 4))
    axes = np.atleast_1d(axes)
    for h, ax in enumerate(axes):
        im = ax.imshow(A[h], aspect="auto", cmap="viridis")
        ax.set(title=f"head {h}", xlabel="key pos", ylabel="query pos")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 3. digit attention
@torch.no_grad()
def digit_attn_maps(model, xte, out, B=256):
    model(xte[:B])
    A   = model.blocks[0].mha.attn.cpu()                      # (B, H, L, L)
    tok = xte[:B].cpu()                                       # (B, L), BOS = 10
    L   = tok.shape[1]

    oh     = F.one_hot(tok, 11).float()[..., :D]             # (B, L, D); BOS row -> all-zero
    causal = torch.tril(torch.ones(L, L), diagonal=-1)        # strict k < q: prefix only

    num = torch.einsum('bhqk,qk,bqi,bkj->hij', A, causal, oh, oh)   # (H, D, D) summed attn
    den = torch.einsum('qk,bqi,bkj->ij',      causal, oh, oh)       # (D, D) pair counts
    M   = num / den.clamp(min=1)
    M   = M / M.sum(-1, keepdim=True).clamp(min=1e-9)               # distribution over key-digit

    fig, axes = plt.subplots(1, M.shape[0], figsize=(4 * M.shape[0], 4))
    axes = np.atleast_1d(axes)
    for h, ax in enumerate(axes):
        im = ax.imshow(M[h], cmap="viridis", vmin=0)
        ax.set(title=f"head {h}", xlabel="key digit", ylabel="query digit")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 4. head-ablation sanity
def head_ablation_sanity(model, H, device, L, K=5):
    mha0 = model.blocks[0].mha

    def probe_r2(tr_in, tr_tg, te_in, te_tg, point="mid0"):
        Xtr = cache_resids(model, tr_in)[point]
        Xte = cache_resids(model, te_in)[point]
        Xtr = Xtr.reshape(-1, Xtr.shape[-1]).numpy()
        Xte = Xte.reshape(-1, Xte.shape[-1]).numpy()
        p = Ridge(alpha=1.0).fit(Xtr, dev_targets(tr_tg))
        return r2_score(dev_targets(te_tg), p.predict(Xte))

    draws = [(*sample_batch(256, L, device=device),
              *sample_batch(256, L, device=device)) for _ in range(K)]

    def condition(ablate):
        mha0.ablate_heads = ablate
        rs = [probe_r2(*d) for d in draws]
        mha0.ablate_heads = []
        return np.mean(rs), np.std(rs)

    print("  head-ablation count-deviation R2 at mid0:")
    for name, ab in [("none", []), ("all", list(range(H)))]:
        m, s = condition(ab)
        print(f"    {name:9s}: {m:.3f} +/- {s:.3f}")


# ----------------------------------------------------------------------------- 5/6. OV circuit
def ov_write_vectors(model, h, H):
    """Per-digit write vectors of head h: how each digit token, read through the
    head's value+output projection, gets written back into the residual stream."""
    blk, mha = model.blocks[0], model.blocks[0].mha
    d_k = mha.lin_v.out_features // H
    with torch.no_grad():
        e  = model.emb.weight[:10]            # (10, 64) digit-token embeddings, no BOS
        e  = blk.norm1(e)                     # pre-LN: what the head actually reads
        v  = mha.lin_v(e).view(10, H, d_k)    # per-head values
        vm = torch.zeros_like(v); vm[:, h] = v[:, h]
        vm = vm.reshape(10, H * d_k)          # head-h only
        w  = vm @ mha.out.weight.T            # (10, 64) per-digit write vectors (bias dropped)
    wn = w / w.norm(dim=1, keepdim=True)
    return w, wn


def ov_geometry_plot(w, wn, h, out):
    C = (wn @ wn.T).cpu()
    fig, ax = plt.subplots()
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(title=f"head {h}: cosine between per-digit write vectors",
           xlabel="digit", ylabel="digit", xticks=range(10), yticks=range(10))
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)

    S = torch.linalg.svdvals(w)
    print("  OV singular values:", S.round(decimals=3).tolist())
    print("  write norms/digit :", w.norm(dim=1).round(decimals=3).tolist())


def ov_loop_closure(model, h, H, w, wn, device, L, out):
    """Fit a count-deviation read-out using ONLY head h, then compare the read-out
    directions to the write directions (cosine) and the per-digit decodability."""
    mha = model.blocks[0].mha
    mha.ablate_heads = [x for x in range(H) if x != h]
    xi, yi = sample_batch(256, L, device=device)
    xj, yj = sample_batch(256, L, device=device)
    Xi = cache_resids(model, xi)["mid0"]; Xi = Xi.reshape(-1, Xi.shape[-1]).numpy()
    Xj = cache_resids(model, xj)["mid0"]; Xj = Xj.reshape(-1, Xj.shape[-1]).numpy()
    probe = Ridge(alpha=1.0).fit(Xi, dev_targets(yi))
    mha.ablate_heads = []

    R  = torch.tensor(probe.coef_, dtype=w.dtype).to(device)
    Rn = R / R.norm(dim=1, keepdim=True)
    print("  read<->write cosine/digit:", (Rn * wn).sum(1).round(decimals=2).tolist())

    pred, true = probe.predict(Xj), dev_targets(yj)
    r2d = np.array([r2_score(true[:, d], pred[:, d]) for d in range(10)])
    print(f"  per-digit R2 (head {h} only):", [round(x, 2) for x in r2d])

    M = (Rn @ wn.T).cpu().numpy()                    # cos(count-readout_i, write_j)
    diag = np.diag(M)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    im = ax[0].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set(title="count readout . digit write  (cosine)", xlabel="write direction (digit j)",
              ylabel="count readout (digit i)", xticks=range(10), yticks=range(10))
    fig.colorbar(im, ax=ax[0], shrink=0.8)

    x = np.arange(10)
    ax[1].bar(x - 0.2, diag, 0.4, label="read<->write cosine", color="#c0464b")
    ax[1].bar(x + 0.2, r2d,  0.4, label=f"per-digit R2 (head {h} only)", color="#3f7fb0")
    ax[1].set(title=f"head {h} alone: aligned and decodable", xlabel="digit",
              ylim=(0, 1), xticks=range(10))
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 7. steering (static)
def steer_digit(model, w, d, device, L, out):
    mha = model.blocks[0].mha
    wd = w[d] / w[d].norm()
    xb, yb = sample_batch(512, L, device=device)

    # calibrate scale at the injection point (mha output)
    cap = {}
    h0 = mha.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    with torch.no_grad():
        model(xb)
    h0.remove()
    sd = (cap["o"] @ wd).std().item()                # natural spread of signal along w_d

    state  = {"a": 0.0}
    handle = mha.register_forward_hook(lambda m, i, o: o + state["a"] * wd)
    mults  = np.linspace(-3, 3, 13)
    P = np.zeros((len(mults), 10))
    with torch.no_grad():
        for k, mlt in enumerate(mults):
            state["a"] = mlt * sd
            p = F.softmax(model(xb), dim=-1)[..., :10]
            P[k] = p.mean(dim=(0, 1)).cpu().numpy()
    handle.remove()

    fig = plt.figure(figsize=(8, 5))
    for dd in range(10):
        plt.plot(mults, P[:, dd], color=("crimson" if dd == d else "0.8"),
                 lw=(2.5 if dd == d else 1), label=(f"digit {d} (steered)" if dd == d else None))
    plt.axvline(0, color="k", ls=":", lw=1)
    plt.xlabel(f"injection along digit-{d} write direction  (x natural SD)")
    plt.ylabel("mean P(next = digit)"); plt.legend()
    plt.title(f"steering the digit-{d} count direction"); plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 8. steering over position
def steer_dynamics(model, w, d, device, L, out):
    mha = model.blocks[0].mha
    wd = w[d] / w[d].norm()
    xb, yb = sample_batch(512, L, device=device)

    cap = {}
    h0 = mha.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    with torch.no_grad():
        model(xb)
    h0.remove()
    sd = (cap["o"] @ wd).std().item()

    state  = {"a": 0.0}
    handle = mha.register_forward_hook(lambda m, i, o: o + state["a"] * wd)
    levels = [-3, -1.5, 0, 1.5, 3]
    P6 = {}
    with torch.no_grad():
        for lv in levels:
            state["a"] = lv * sd
            p = F.softmax(model(xb), dim=-1)[..., d]     # (B, L) = P(next=d) per position
            P6[lv] = p.mean(0).cpu().numpy()
    handle.remove()

    fig = plt.figure(figsize=(9, 5))
    for lv in levels:
        plt.plot(range(L), P6[lv], lw=(2.6 if lv == 0 else 1.6),
                 color=("k" if lv == 0 else None),
                 label=("unsteered" if lv == 0 else f"{lv:+.1f} SD"))
    plt.xlabel("position in sequence (t)"); plt.ylabel(f"mean P(next = {d})")
    plt.title(f"digit-{d} steering, resolved over the sequence"); plt.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 9. greedy steering
def steer_greedy(model, w, d, device, L, out, levels=(-3, -1.5, 0, 1.5, 3)):
    """Same as steer_dynamics, but the model GENERATES its own sequence greedily
    (closed loop) while steered along digit-d's write direction. We record the
    probability the model assigns to digit d at each generated position."""
    mha = model.blocks[0].mha
    bos_id = model.out.out_features - 1
    wd = w[d] / w[d].norm()

    # calibrate the natural spread along wd on real sequences (same as static case)
    xb, _ = sample_batch(512, L, device=device)
    cap = {}
    h0 = mha.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    with torch.no_grad():
        model(xb)
    h0.remove()
    sd = (cap["o"] @ wd).std().item()

    state  = {"a": 0.0}
    handle = mha.register_forward_hook(lambda m, i, o: o + state["a"] * wd)
    P = {}
    with torch.no_grad():
        for lv in levels:
            state["a"] = lv * sd
            seq = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
            probs = []
            for _ in range(L):
                logits = model(seq)[:, -1, :].clone()
                probs.append(F.softmax(logits, dim=-1)[0, d].item())  # P(next = d)
                logits[:, bos_id] = float("-inf")                     # never emit BOS
                nxt = logits.argmax(dim=-1)                           # greedy
                seq = torch.cat([seq, nxt[:, None]], dim=1)
            P[lv] = np.array(probs)
    handle.remove()

    fig = plt.figure(figsize=(9, 5))
    for lv in levels:
        plt.plot(range(L), P[lv], lw=(2.6 if lv == 0 else 1.6),
                 color=("k" if lv == 0 else None),
                 label=("unsteered" if lv == 0 else f"{lv:+.1f} SD"))
    plt.xlabel("generated position t"); plt.ylabel(f"P(next = {d})")
    plt.title(f"digit-{d} steering under greedy decoding"); plt.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- 9b. emission counts
def steer_emission_counts(model, w, d, device, L, n, out,
                          levels=(-3, -1.5, 0, 1.5, 3), B=512, temperature=1.0):
    """How many times does digit d actually get emitted, as a function of steering?

    Greedy is deterministic (n=1 per condition), so here we use ancestral sampling
    (T=temperature) -- the decoding that matches how the model was trained -- and
    draw B closed-loop rollouts per steering level to get a real distribution of
    per-sequence digit-d emission counts. Greedy is overlaid as a reference point.
    """
    mha = model.blocks[0].mha
    bos_id = model.out.out_features - 1
    wd = w[d] / w[d].norm()

    xb, _ = sample_batch(512, L, device=device)
    cap = {}
    h0 = mha.register_forward_hook(lambda m, i, o: cap.__setitem__("o", o.detach()))
    with torch.no_grad():
        model(xb)
    h0.remove()
    sd = (cap["o"] @ wd).std().item()

    state  = {"a": 0.0}
    handle = mha.register_forward_hook(lambda m, i, o: o + state["a"] * wd)

    def rollout(batch, greedy):
        seq = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(L):
            logits = model(seq)[:, -1, :].clone()
            logits[:, bos_id] = float("-inf")
            if greedy:
                nxt = logits.argmax(dim=-1)
            else:
                p = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(p, 1).squeeze(-1)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        gen = seq[:, 1:]
        return (gen == d).sum(dim=1).cpu().numpy()      # (batch,) emission counts

    sampled, greedy_cnt = {}, {}
    with torch.no_grad():
        for lv in levels:
            state["a"] = lv * sd
            sampled[lv] = rollout(B, greedy=False)
            greedy_cnt[lv] = int(rollout(1, greedy=True)[0])
    handle.remove()

    pos = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([sampled[lv] for lv in levels], positions=pos, widths=0.6, showmeans=True)
    ax.plot(pos, [greedy_cnt[lv] for lv in levels], "rD", ms=8, zorder=5,
            label="greedy (deterministic)")
    ax.axhline(n, ls="--", c="gray", label=f"correct count n={n}")
    ax.set_xticks(pos); ax.set_xticklabels([f"{lv:+.1f}" for lv in levels])
    ax.set(xlabel="steering level  (x SD along digit-%d write direction)" % d,
           ylabel=f"# of digit-{d} emissions per sequence",
           title=f"digit-{d} emission count vs steering (sampled T={temperature}, B={B})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)

    print(f"  level | mean+/-std (sampled) | greedy")
    for lv in levels:
        c = sampled[lv]
        print(f"  {lv:>+5.1f} | {c.mean():5.2f} +/- {c.std():4.2f}        | {greedy_cnt[lv]}")
    print("saved:", out)


# ----------------------------------------------------------------------------- 10. OOD count forcing
def ood_count_forcing(model, device, L, d, n, out):
    """Probe: fit a consensus count-d read-out on the FULL residual (all heads) at
    mid0, then force that read to specified count values (in- and out-of-distribution)
    and watch P(next = d) across the sequence vs the real remaining-budget curve."""
    mha = model.blocks[0].mha
    tvec = torch.arange(L, device=device).float()

    # read-out fit on the full residual (all heads) -> the consensus count-d direction
    xc, yc = sample_batch(512, L, device=device)
    Xc = cache_resids(model, xc)["mid0"]; Xc = Xc.reshape(-1, Xc.shape[-1])
    fp = Ridge(alpha=1.0).fit(Xc.numpy(), dev_targets(yc))
    Rd = torch.tensor(fp.coef_[d], dtype=torch.float32, device=device)
    bd = float(fp.intercept_[d]); g = Rd.dot(Rd)

    x1, y1 = sample_batch(1, L, device=device)
    cd = counts_from_targets(y1)[0, :, d].to(device).float()
    true_p = ((n - cd).clamp(min=0) / (L - tvec)).cpu().numpy()

    forced = {"c": None}

    def install(m, i, o):
        if forced["c"] is None:
            return o
        cur = (o @ Rd) + bd                       # current read deviation (approx)
        tgt = forced["c"] - tvec / D              # target deviation for forced count
        return o + ((tgt - cur) / g).unsqueeze(-1) * Rd   # set consensus read to target
    handle = mha.register_forward_hook(install)

    runs = {}
    with torch.no_grad():
        for c in [None, 0, 5, 40, -40]:
            forced["c"] = c
            runs[c] = F.softmax(model(x1), dim=-1)[0, :, d].cpu().numpy()
    handle.remove()

    fig = plt.figure(figsize=(9, 5))
    plt.plot(range(L), true_p, "k--", lw=2, label="real budget")
    for c, lab in [(None, "unsteered"), (0, "count=0"), (5, "count=5"),
                   (40, "count=40 (OOD)"), (-40, "count=-40 (OOD)")]:
        plt.plot(range(L), runs[c], lw=2, label=lab)
    plt.xlabel("position t"); plt.ylabel(f"P(next = {d})"); plt.legend()
    plt.title(f"forcing the count-{d} read to in/out-of-distribution values")
    plt.tight_layout()
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved:", out)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    # checkpoints/best.pt is the stale pre-GELU-fix model (0.58-nat gap); the corrected
    # near-Bayes L=50 model lives in checkpoints_count50/best.pt (see scripts/train_count50.py).
    ap.add_argument("--ckpt", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--outdir", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/llm_images")
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    model, ck = load_model(args.ckpt, device)
    L = ck["args"]["seq_length"]
    H = ck["model_args"]["num_heads"]
    print(f"checkpoint step={ck['step']}  L={L}  D={D}  H={H}  device={args.device}\n")

    # a shared held-out batch for the attention-pattern figures
    xte, yte = sample_batch(args.batch, L, device=device)

    print("[1] count probe across residual stream")
    count_probe(model, ck, args.batch, L, device, outdir / "count_probe_split.png", args.alpha)

    print("\n[2] block-0 attention maps")
    attn_head_maps(model, xte, outdir / "block0_heads.png")

    print("\n[3] block-0 digit attention")
    digit_attn_maps(model, xte, outdir / "block0_digit_attn.png", B=min(args.batch, 256))

    print("\n[4] head-ablation sanity")
    head_ablation_sanity(model, H, device, L)

    print(f"\n[5] head-{HEAD} OV write geometry")
    w, wn = ov_write_vectors(model, HEAD, H)
    ov_geometry_plot(w, wn, HEAD, outdir / f"head{HEAD}_OV_geometry.png")

    print(f"\n[6] head-{HEAD} loop closure (read<->write)")
    ov_loop_closure(model, HEAD, H, w, wn, device, L, outdir / f"head{HEAD}_loop_closure.png")

    print(f"\n[7] steering along digit-{STEER_DIGIT} write direction")
    steer_digit(model, w, STEER_DIGIT, device, L, outdir / "steer_digit.png")

    print(f"\n[8] steering dynamics over position (teacher-forced)")
    steer_dynamics(model, w, STEER_DIGIT, device, L, outdir / "steer_dynamics.png")

    print("\n[9] steering under greedy decoding, one plot per digit")
    for dd in range(D):
        steer_greedy(model, w, dd, device, L, outdir / f"steer_greedy_digit{dd}.png")

    n = L // D
    print(f"\n[9b] digit-{STEER_DIGIT} emission count vs steering (sampled)")
    steer_emission_counts(model, w, STEER_DIGIT, device, L, n,
                          outdir / "steer_emission_counts.png")

    print(f"\n[10] OOD count-forcing probe (digit {STEER_DIGIT}, n={n})")
    ood_count_forcing(model, device, L, STEER_DIGIT, n, outdir / "ood_steering.png")

    print("\nall figures written to", outdir)


if __name__ == "__main__":
    main()
