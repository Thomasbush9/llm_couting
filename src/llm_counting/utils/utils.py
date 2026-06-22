import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
D = 10

def counts_from_targets(targets):
    """Exclusive prefix counts c(t): (B, L, D) — same alignment as true_conditional."""
    oh = F.one_hot(targets, num_classes=D).float()
    return torch.cumsum(oh, dim=1) - oh

@torch.no_grad()
def cache_resids(model, inputs):
    """Hook residual points + sublayer deltas, return the reconstructed stream."""
    raw = {}
    def save(name):
        def hook(m, i, o): raw[name] = o.detach().cpu().float()
        return hook

    handles = [model.emb.register_forward_hook(save("emb"))]
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(save(f"block{i}")))         # resid_post
        handles.append(blk.mha.register_forward_hook(save(f"attn_delta{i}")))
        handles.append(blk.mlp.register_forward_hook(save(f"mlp_delta{i}")))
    handles.append(model.norm.register_forward_hook(save("final")))

    model.eval(); model(inputs)
    for h in handles: h.remove()

    pts = {
        "emb":   raw["emb"],                          # resid entering block 0  (control)
        "mid0":  raw["emb"]   + raw["attn_delta0"],   # after block-0 ATTENTION
        "post0": raw["block0"],                        # after block-0 MLP  (== resid entering block 1)
        "mid1":  raw["block0"] + raw["attn_delta1"],  # after block-1 ATTENTION
        "post1": raw["block1"],                        # after block-1 MLP
        "final": raw["final"],                         # post final norm
    }
    # verify the delta reconstruction is faithful
    assert torch.allclose(pts["post0"], pts["mid0"] + raw["mlp_delta0"], atol=1e-4)
    return pt
