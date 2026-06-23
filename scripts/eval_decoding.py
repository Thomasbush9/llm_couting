"""Evaluate decoding strategies on a trained checkpoint.

Compares teacher-forced CE (the current validation signal) against
free-running autoregressive generation under greedy / sampling / constrained
decoding, reporting how often the model produces a *valid* balanced sequence.
"""
import sys, argparse
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_counting.model.model import Transformer
from llm_counting.train.trainer import sample_batch, true_conditional, entropy_floor


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["model_args"]); cfg.pop("VOCAB", None)
    model = Transformer(**cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def validity_stats(gen, D, n):
    """gen: (B, L) digit ids. Returns dict of validity metrics."""
    B = gen.size(0)
    counts = torch.zeros(B, D, dtype=torch.long, device=gen.device)
    counts.scatter_add_(1, gen, torch.ones_like(gen))
    valid = (counts == n).all(dim=1)                 # exact balanced permutation
    over = (counts - n).clamp(min=0)                 # how far each digit exceeds cap
    return {
        "valid_rate": valid.float().mean().item(),
        "mean_overused_digits": (counts > n).sum(1).float().mean().item(),
        "max_overflow": over.max().item(),
        "mean_overflow": over.sum(1).float().mean().item(),
    }


@torch.no_grad()
def teacher_forced_ce(model, batch_size, L, device, n_batches=20):
    tot = 0.0
    for _ in range(n_batches):
        inp, tgt = sample_batch(batch_size, L, device=device)
        logits = model(inp)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()
    return tot / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints/best.pt")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, ck = load_model(args.ckpt, device)
    L = ck["args"]["seq_length"]
    D = ck["model_args"]["out_dim"] - 1
    n = L // D
    print(f"checkpoint step={ck['step']}  L={L}  D={D}  n(per-digit)={n}  batch={args.batch}\n")

    # reference: teacher-forced CE vs Bayes floor
    ce = teacher_forced_ce(model, args.batch, L, device)
    floor = entropy_floor(seq_len=L, batch_size=args.batch, device=args.device)
    print(f"[teacher-forced]  CE={ce:.4f}  floor={floor:.4f}  gap={ce-floor:.4f}\n")

    # Free-running decoding only -- the model gets no hint of the count cap.
    # (Constrained decoding is excluded: masking exhausted digits injects the
    #  task's ground-truth constraint into the decoder, so 100% validity is
    #  guaranteed by construction and measures the mask, not the model.)
    configs = [
        ("greedy        ", dict(mode="greedy")),
        ("sample T=1.0   ", dict(mode="sample", temperature=1.0)),
        ("sample T=0.7   ", dict(mode="sample", temperature=0.7)),
        ("sample T=0.5   ", dict(mode="sample", temperature=0.5)),
    ]
    print(f"{'strategy':<24} {'valid%':>7} {'overused_d':>11} {'mean_ovf':>9} {'max_ovf':>8}")
    print("-" * 64)
    for name, kw in configs:
        gen = model.generate(args.batch, L, n_per_digit=n, device=args.device, **kw)
        s = validity_stats(gen, D, n)
        print(f"{name:<24} {100*s['valid_rate']:>6.1f}% {s['mean_overused_digits']:>11.2f} "
              f"{s['mean_overflow']:>9.2f} {s['max_overflow']:>8d}")


if __name__ == "__main__":
    main()
