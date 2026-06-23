"""Retrain the counting transformer with the MLP nonlinearity fixed.

Replicates the original checkpoint's config (batch 256, 10k steps, L=50, lr 3e-4,
wd 0.01) so the only change vs best.pt is the corrected MLP (input -> GELU -> output).
Saves to a separate checkpoint dir to leave the original best.pt intact.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_counting.train.trainer import Train, TrainingArgs, ModelArgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir",
                    default="/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_relu")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-steps", type=int, default=10000)
    args = ap.parse_args()

    targs = TrainingArgs(
        batch_size=256, num_steps=args.num_steps, seq_length=50, device=args.device,
        lr=3e-4, val_every=200, log_every=50, weight_decay=0.01,
        checkpoint_every_step=5000, save_checkpoints=True, probe_size=4,
    )
    margs = ModelArgs(
        input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64,
        num_heads=4, causal=True, max_len=512, num_blocks=2, VOCAB=11,
    )

    trainer = Train(targs, margs)
    trainer.ckpt_dir = Path(args.ckpt_dir)
    trainer.ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"entropy floor (Bayes CE) = {trainer.entropy_floor:.4f}")
    trainer.training_loop()
    print(f"\nDONE  best_val_CE={trainer.best_val:.4f}  "
          f"gap={trainer.best_val - trainer.entropy_floor:.4f}")
    print("best checkpoint:", trainer.ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
