import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Any, Dict
import math
from einops import rearrange


from IPython.terminal.interactiveshell import Bool


class RotaryEmbeddings(nn.Module):
  def __init__(self, dim, max_len=512, base=10_000):
    super().__init__()
    assert dim % 2 == 0, "RoPE needs an even per-head dim"
    inv_freq = 1.0 / (base** (torch.arange(0, dim, 2).float() /dim))
    pos = torch.arange(max_len).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    self.register_buffer('cos', emb.cos()[None, None], persistent=False)
    self.register_buffer('sin', emb.sin()[None, None], persistent=False)

  def forward(self, seq_len):
    return self.cos[..., :seq_len, :], self.sin[..., :seq_len, :]

def rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, cos, sin):
  return x * cos + rotate_half(x) * sin


class AttentionModule(nn.Module):
  def __init__(self, in_dim, hidden, num_heads=None, causal:bool=True, max_len:int=512):
    super().__init__()
    self.hidden = hidden
    self.causal = causal
    self.num_heads = num_heads
    if num_heads is not None:
      self.d_k = self.hidden // num_heads
    else:
      self.d_k = hidden
    self.rope = RotaryEmbeddings(self.d_k, max_len=max_len)
    self.lin_q = nn.Linear(in_dim, hidden)
    self.lin_k = nn.Linear(in_dim, hidden)
    self.lin_v = nn.Linear(in_dim, hidden)
    self.out = nn.Linear(hidden, in_dim)
    # pre build mask
    if self.causal:
      self.register_buffer('mask', torch.triu(torch.full((1,1,max_len,max_len), -torch.inf), diagonal=1))
    else:
      self.register_buffer('mask', torch.zeros((1, 1, max_len, max_len)))
    self.attn = None
    self.ablate_heads = []

  def forward(self, x: torch.Tensor):
    # from [BATCH, SEQ, HIDDEN-> BATCH, NUM_HEADS, SEQ, HIDDEN / NUM_HEADS]
    batch, seq_len, input_dim = x.shape
    cos, sin = self.rope(seq_len)
    if self.num_heads:
      Q = rearrange(self.lin_q(x), 'b s (n h) -> b n s h', n=self.num_heads, h=self.d_k)
      Q = apply_rotary(Q, cos, sin)
      K = rearrange(self.lin_k(x), 'b s (n h) -> b n s h', n=self.num_heads, h=self.d_k)
      K = apply_rotary(K, cos, sin)
      V = rearrange(self.lin_v(x), 'b s (n h) -> b n s h', n=self.num_heads, h=self.d_k)
            # score
      score = torch.einsum('bnqh, bnkh-> bnqk', Q, K) / math.sqrt(self.d_k)
      score = score + self.mask[..., :seq_len, :seq_len]
      score = torch.softmax(score, dim=-1)
      self.attn = score.detach()
      out = torch.einsum('bnsv, bnqs->bnqv', V, score)
      #apply ablations
      if self.ablate_heads:
        out[:, self.ablate_heads] = 0.0
      out = rearrange(out, 'b n s h -> b s (n h)', n=self.num_heads, h=self.d_k)
      out = self.out(out)
    else:
      cos, sin = cos.squeeze(dim=1), sin.squeeze(dim=1)
      Q = self.lin_q(x)
      Q = apply_rotary(Q, cos, sin)
      K = self.lin_k(x)
      K = apply_rotary(K, cos, sin)
      V = self.lin_v(x)
      score = torch.einsum('bqh, bkh-> bqk', Q, K) / math.sqrt(self.d_k)
      score = score + self.mask[..., :seq_len, :seq_len].squeeze(dim=1)
      score = torch.softmax(score, dim=-1)
      self.attn = score.detach()
      out = torch.einsum('bsv, bqs->bqv', V, score)
      out = self.out(out)
    return out


class MLP(nn.Module):
  def __init__(self, in_dim, hidden):
    super().__init__()
    self.input = nn.Linear(in_dim, hidden)
    self.output = nn.Linear(hidden, in_dim)
  def forward(self, x:torch.Tensor):
    out = self.input(x)
    out = self.output(x)
    return out

class TransformerBlock(nn.Module):
  def __init__(self,input_dim, attn_dim, hidden_dim, causal:bool=True, num_heads:int=4, max_len:int=512):
    super().__init__()
    self.mha = AttentionModule(in_dim=input_dim,
                               hidden=attn_dim,
                               causal=causal,
                               max_len=max_len,
                               num_heads=num_heads)
    self.mlp = MLP(in_dim=input_dim, hidden=hidden_dim)
    self.norm1 = nn.LayerNorm(input_dim)
    self.norm2 = nn.LayerNorm(input_dim)

  def forward(self, x:torch.Tensor):
    x = x + self.mha(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x

#TODO: add decoding strategy 
class Transformer(nn.Module):
  def __init__(self, input_dim, out_dim, attn_dim, hidden_dim, num_heads=4, causal=True, max_len=512,num_blocks:int=2,VOCAB:int=11):
    super().__init__()
    self.emb = nn.Embedding(VOCAB, input_dim)
    self.blocks = nn.Sequential(*[
        TransformerBlock(
            input_dim=input_dim,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            causal=causal,
            max_len=max_len
        )
        for i in range(num_blocks)
    ])
    self.norm = nn.LayerNorm(input_dim)
    self.out = nn.Linear(input_dim, out_dim)

  def forward(self, x:torch.Tensor):
    x = self.emb(x)
    x = self.blocks(x)
    x = self.norm(x)
    return self.out(x)
