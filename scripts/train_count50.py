"""Train a fresh L=50 balanced-counting model with the CORRECTED GELU MLP.

The old best.pt was trained before the MLP nonlinearity fix (dead input layer);
loaded into the now-GELU architecture it has a 0.58-nat gap. This trains a clean
model to a low gap and saves it in the same checkpoint format the probing scripts
expect (keys: step, model, args, model_args).
"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from llm_counting.model.model import Transformer
from llm_counting.train.trainer import sample_batch, true_conditional, entropy_floor

device = "cuda"; L = 50; D = 10; VOCAB = 11
CK = Path("/n/holylfs06/LABS/bsabatini_lab/Everyone/tbush/checkpoints_count50"); CK.mkdir(parents=True, exist_ok=True)
MODEL_ARGS = dict(input_dim=64, out_dim=11, attn_dim=64, hidden_dim=64, num_heads=4,
                  causal=True, max_len=512, num_blocks=2)


@torch.no_grad()
def val(model, floor, nb=20, bs=256):
    ce = 0.0; kl_t = torch.zeros(L)
    for _ in range(nb):
        x, y = sample_batch(bs, L, device=device)
        lg = model(x)
        ce += F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1)).item()
        lg2 = lg.clone(); lg2[..., -1] = float("-inf")
        pm = torch.softmax(lg2, -1)[..., :D].cpu(); pt = true_conditional(y.cpu(), VOCAB=VOCAB)
        kl_t += (torch.xlogy(pt, pt) - torch.xlogy(pt, pm.clamp(min=1e-12))).sum(-1).mean(0)
    return ce / nb, ce / nb - floor, (kl_t / nb)


def main(steps=16000, bs=256):
    t0 = time.time()
    floor = entropy_floor(seq_len=L, batch_size=256, device=device, n_batches=50)
    print(f"floor={floor:.4f}", flush=True)
    m = Transformer(**MODEL_ARGS).to(device)
    opt = optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    best = float("inf"); best_state = None; best_step = 0
    for s in range(steps):
        m.train(); x, y = sample_batch(bs, L, device=device)
        opt.zero_grad(); F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1)).backward(); opt.step()
        if s % 400 == 0:
            m.eval(); ce, gap, _ = val(m, floor)
            if ce < best:
                best, best_step = ce, s
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            print(f"step {s:5d} | val_ce {ce:.4f} | gap {gap:.4f}", flush=True)
    m.load_state_dict(best_state); m.eval()
    ce, gap, kl_t = val(m, floor, nb=40)
    torch.save({"step": best_step, "model": m.state_dict(),
                "args": {"seq_length": L}, "model_args": dict(MODEL_ARGS, VOCAB=VOCAB)}, CK / "best.pt")
    json.dump({"floor": floor, "val_ce": ce, "gap": gap, "best_step": best_step,
               "kl_by_pos": [round(float(v), 4) for v in kl_t], "minutes": (time.time() - t0) / 60},
              open(CK / "results.json", "w"), indent=2)
    print(f"FINAL gap={gap:.4f} ce={ce:.4f}", flush=True)
    print("KL last 10 pos:", [round(float(v), 3) for v in kl_t[-10:]], flush=True)
    print(f"saved {CK/'best.pt'}\nTRAIN DONE ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
