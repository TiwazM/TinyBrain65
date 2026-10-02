"""sidlm transformer.

Decoder-only, pre-norm, RMSNorm everywhere, rotary positions, QK-norm for
stable training at a high learning rate, SwiGLU feed-forward, no biases,
input and output embeddings tied.  ``ModelConfig()`` gives 5.88M parameters
with the 925-token vocabulary and a 4096-token context.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 1024


@dataclass
class ModelConfig:
    vocab: int = VOCAB
    n_ctx: int = 4096
    d_model: int = 256
    n_layer: int = 8
    n_head: int = 4
    d_ff: int = 576
    dropout: float = 0.0
    rope_base: float = 10000.0

    def param_count(self) -> int:
        d, h = self.d_model, self.d_ff
        head_dim = d // self.n_head
        per_layer = 4 * d * d + 3 * d * h + 2 * d + 2 * head_dim
        return self.vocab * d + self.n_layer * per_layer + d


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def _rope_tables(n_ctx: int, head_dim: int, base: float):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(n_ctx, dtype=torch.float32)
    freqs = torch.outer(t, inv)                       # (n_ctx, head_dim/2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (b, h, t, d); cos/sin: (t, d/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack((y1, y2), dim=-1).flatten(-2)


class KVCache:
    """Preallocated per-layer key/value buffers for incremental decoding."""

    def __init__(self, cfg: ModelConfig, batch: int, device, dtype):
        head_dim = cfg.d_model // cfg.n_head
        shape = (batch, cfg.n_head, cfg.n_ctx, head_dim)
        self.k = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.v = [torch.empty(shape, device=device, dtype=dtype) for _ in range(cfg.n_layer)]
        self.length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        t = k.shape[2]
        self.k[layer][:, :, self.length:self.length + t] = k
        self.v[layer][:, :, self.length:self.length + t] = v
        end = self.length + t
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, t: int) -> None:
        self.length += t


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.d_model % cfg.n_head:
            raise ValueError("d_model must divide by n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                cache: Optional[KVCache] = None, layer: int = 0) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim).unbind(2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.append(layer, k, v)
            causal = t > 1
        else:
            causal = True
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0):
        x = x + self.attn(self.norm1(x), cos, sin, cache, layer)
        return x + self.ff(self.norm2(x))


class SidLM(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.embed = nn.Embedding(c.vocab, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        cos, sin = _rope_tables(c.n_ctx, c.d_model // c.n_head, c.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        scale = 0.02 / math.sqrt(2 * c.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, std=scale)
            nn.init.normal_(block.ff.down.weight, std=scale)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[KVCache] = None, return_hidden: bool = False):
        b, t = idx.shape
        start = cache.length if cache is not None else 0
        if start + t > self.cfg.n_ctx:
            raise ValueError("sequence longer than the context window")
        cos = self.rope_cos[start:start + t]
        sin = self.rope_sin[start:start + t]
        ew = self.embed.weight
        x = self.drop(F.embedding(idx, fake_quant(ew,127) if getattr(self,'qat',False) else ew))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, cache, i)
        if cache is not None:
            cache.advance(t)
        x = self.norm(x)
        logits = F.linear(x, fake_quant(ew,7) if getattr(self,'qat',False) else ew)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        # The optional training representation adds no weights or inference work.
        return (logits, loss, x) if return_hidden else (logits, loss)

    def new_cache(self, batch: int = 1) -> KVCache:
        p = next(self.parameters())
        return KVCache(self.cfg, batch, p.device, p.dtype)

    # -- persistence ---------------------------------------------------------
    def save(self, path, extra: Optional[dict] = None) -> None:
        torch.save({"config": asdict(self.cfg), "state": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "SidLM":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ModelConfig(**blob["config"]))
        model.load_state_dict(blob["state"])
        return model

def fake_quant(w,qmax):
    scale=w.detach().abs().amax(-1,keepdim=True).clamp_min(1e-8)/qmax
    rounded=(w/scale).round().clamp(-qmax,qmax)*scale
    return w+(rounded-w).detach()

class Int4Weight(nn.Module):
    def forward(self,w):return fake_quant(w,7)

def enable_qat(model):
    from torch.nn.utils import parametrize
    if getattr(model,'qat',False):return
    for module in model.modules():
        if isinstance(module,nn.Linear):parametrize.register_parametrization(module,'weight',Int4Weight())
    model.qat=True

def plain_state(model):
    # Save unrounded trainable weights; the exporter applies the same quantizer.
    return {k.replace('.parametrizations.weight.original','.weight'):v.detach().cpu().clone() for k,v in model.state_dict().items()}
