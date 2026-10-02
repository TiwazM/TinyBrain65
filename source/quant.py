"""Quantize a sidlm checkpoint and export it as a flat weight blob for the
MEGA65 runtime (``sidlm_core.c``), format version 2.

Numerics (shared with the C code):
  * linear weights   int4 symmetric, values -7..7, one float scale per row.
                     Stored one weight per byte as the *index* value+8 (1..15)
                     so the 6502 kernels can use the byte directly as a table /
                     bucket index; each row also stores the sum of its int values.
  * embedding        int8 per row with a float scale (input lookup)
  * output head      the same embedding rows re-quantized to int4 (tied weights)
  * norm weights     float32
  * RoPE tables      int16 Q14 cos/sin per position
  * K / V cache      int4 with one static float scale per layer/head,
                     calibrated on songs (10 % headroom)

    python -m mega65.quant sidlm/checkpoints/mini.pt mega65/build/mini.bin
    python -m mega65.quant sidlm/checkpoints/mini.pt --eval        # val loss fp32 vs quantized
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import struct
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import SidLM, ModelConfig, _rope_tables  # noqa: E402

MAGIC = b"SIDM"
VERSION = 2
QMAX = 7                     # int4 symmetric range -7..7
HEADROOM = 1.1               # static K/V scales: calibrated peak x 1.1


def quantize_rows(w: torch.Tensor, qmax: int, group: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """w: (rows, cols) -> (int8 codes (rows, cols), float32 scales (rows, cols // group))."""
    rows, cols = w.shape
    g = cols if group == 0 else group
    if cols % g:
        raise ValueError(f"{cols} inputs not divisible by group {g}")
    wg = w.detach().float().cpu().reshape(rows, cols // g, g)
    scale = wg.abs().amax(dim=-1) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(wg / scale[..., None]), -qmax, qmax)
    return q.reshape(rows, cols).to(torch.int8).numpy(), scale.float().numpy()


def dequantize(q: np.ndarray, scale: np.ndarray) -> torch.Tensor:
    rows, cols = q.shape
    g = cols // scale.shape[1]
    wq = torch.from_numpy(q.astype(np.float32)).reshape(rows, cols // g, g)
    return (wq * torch.from_numpy(scale)[..., None]).reshape(rows, cols)


def index_bytes(q: np.ndarray) -> bytes:
    """(rows, cols) int8 in -7..7 -> one byte per weight holding value + 8 (1..15)."""
    return (q.astype(np.int16) + 8).astype(np.uint8).tobytes()


def row_meta(q: np.ndarray, scale: np.ndarray) -> bytes:
    """Per row: float32 scale, int16 sum of the int values, int16 zero pad (8 bytes)."""
    sums = q.astype(np.int32).sum(axis=1).astype("<i2")
    meta = np.zeros(q.shape[0], dtype=[("s", "<f4"), ("sum", "<i2"), ("pad", "<i2")])
    meta["s"] = scale.reshape(-1)
    meta["sum"] = sums
    return meta.tobytes()


def fake_quant_model(model: SidLM, group: int = 0) -> SidLM:
    """Copy of the model with weights replaced by their dequantized values."""
    m = copy.deepcopy(model).float()
    dev = next(m.parameters()).device
    with torch.no_grad():
        q, s = quantize_rows(m.embed.weight, 127)
        m.embed.weight.copy_(dequantize(q, s).to(dev))
        for blk in m.blocks:
            for lin in (blk.attn.qkv, blk.attn.out, blk.ff.gate, blk.ff.up, blk.ff.down):
                q, s = quantize_rows(lin.weight, QMAX, group)
                lin.weight.copy_(dequantize(q, s).to(dev))
    return m


@torch.no_grad()
def calibrate_kv(model: SidLM, sequences, device) -> Tuple[np.ndarray, np.ndarray]:
    """Peaks per layer/head over the calibration sequences -> (k_peak, v_peak), each (n_layer, n_head).

    K is measured after the QK-norm as the largest RoPE pair magnitude, which bounds
    every component after the rotation; V straight from the qkv projection."""
    cfg = model.cfg
    hd = cfg.d_model // cfg.n_head
    k_peak = torch.zeros(cfg.n_layer, cfg.n_head, device=device)
    v_peak = torch.zeros(cfg.n_layer, cfg.n_head, device=device)
    hooks = []
    for li, blk in enumerate(model.blocks):
        def v_hook(mod, inp, out, li=li):
            b, t, _ = out.shape
            v = out.view(b, t, 3, cfg.n_head, hd)[:, :, 2]
            v_peak[li] = torch.maximum(v_peak[li], v.abs().amax(dim=(0, 1, 3)))

        def k_hook(mod, inp, out, li=li):              # out: (b, t, n_head, hd) after k_norm
            pair = torch.sqrt(out[..., 0::2] ** 2 + out[..., 1::2] ** 2)
            k_peak[li] = torch.maximum(k_peak[li], pair.amax(dim=(0, 1, 3)))
        hooks.append(blk.attn.qkv.register_forward_hook(v_hook))
        hooks.append(blk.attn.k_norm.register_forward_hook(k_hook))
    for ids in sequences:
        model(torch.tensor([ids], dtype=torch.long, device=device))
    for h in hooks:
        h.remove()
    return k_peak.cpu().numpy(), v_peak.cpu().numpy()


def export(model: SidLM, path: Path, k_peak: np.ndarray, v_peak: np.ndarray) -> dict:
    cfg = model.cfg
    hd = cfg.d_model // cfg.n_head
    out = bytearray()
    out += MAGIC
    out += struct.pack("<HHHHHHHHBB", VERSION, cfg.vocab, cfg.d_model, cfg.n_layer, cfg.n_head,
                       cfg.d_ff, cfg.n_ctx, 0, 4, 0)

    def f32(t): out.extend(np.asarray(t, dtype=np.float32).tobytes())

    def lin(w):
        q, s = quantize_rows(w, QMAX)
        out.extend(index_bytes(q)); out.extend(row_meta(q, s))

    sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    q, s = quantize_rows(sd["embed.weight"], 127)
    out.extend(q.astype(np.int8).tobytes()); f32(s)                 # input embedding
    lin(sd["embed.weight"])                                          # tied output head, int4
    f32(sd["norm.weight"])
    cos, sin = _rope_tables(cfg.n_ctx, hd, cfg.rope_base)
    out.extend(np.clip(np.round(cos.numpy() * 16384), -32767, 32767).astype("<i2").tobytes())
    out.extend(np.clip(np.round(sin.numpy() * 16384), -32767, 32767).astype("<i2").tobytes())
    for li in range(cfg.n_layer):
        p = f"blocks.{li}."
        f32(sd[p + "norm1.weight"]); f32(sd[p + "attn.q_norm.weight"]); f32(sd[p + "attn.k_norm.weight"])
        f32(sd[p + "norm2.weight"])
        f32(k_peak[li] * HEADROOM / QMAX)        # static int4 K scale
        f32(v_peak[li] * HEADROOM / QMAX)        # static int4 V scale
        lin(sd[p + "attn.qkv.weight"]); lin(sd[p + "attn.out.weight"])
        lin(sd[p + "ff.gate.weight"]); lin(sd[p + "ff.up.weight"]); lin(sd[p + "ff.down.weight"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    n_lin = sum(w.numel() for k, w in sd.items() if k.endswith("weight") and w.dim() == 2 and "embed" not in k)
    return {"bytes": len(out), "linear_params": n_lin, "embed_params": sd["embed.weight"].numel(),
            "kv_cache_bytes": cfg.n_layer * cfg.n_ctx * 2 * cfg.d_model}


def val_sequences(cfg: ModelConfig, n: int, seed: int = 5):
    from sidlm.dataset import build_examples
    from sidlm.train import sequence_for
    rng = random.Random(seed)
    ex = [e for e in build_examples(200, cfg.n_ctx, log=lambda *_: None) if e.split == "val"]
    rng.shuffle(ex)
    return [sequence_for(e, rng, fixed=True, max_len=cfg.n_ctx) for e in ex[:n]]


@torch.no_grad()
def val_loss(model: SidLM, items, device) -> float:
    total, count = 0.0, 0
    for ids, plen in items:
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        y[:, : plen - 1] = -100
        _, loss = model(x, y)
        n = int((y != -100).sum())
        total += float(loss) * n; count += n
    return total / count


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--eval", action="store_true", help="compare val loss fp32 / int4 linear weights (torch)")
    ap.add_argument("--calib", type=int, default=24, help="calibration songs for the K/V scales")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SidLM.load(args.ckpt, map_location=device).to(device).eval()
    items = val_sequences(model.cfg, max(args.calib, 64 if args.eval else args.calib))
    if args.eval:
        print(f"fp32            val loss {val_loss(model, items[:64], device):.4f}")
        fq = fake_quant_model(model).to(device).eval()
        print(f"int4 weights    val loss {val_loss(fq, items[:64], device):.4f}   (K/V/head effects: see run_pc --compare)")
    if args.out:
        k_peak, v_peak = calibrate_kv(model, [ids for ids, _ in items[: args.calib]], device)
        info = export(model, args.out, k_peak, v_peak)
        print(f"wrote {args.out} ({info['bytes'] / 1e6:.2f} MB): {info['linear_params'] / 1e6:.2f}M int4 linear params, "
              f"{info['embed_params'] / 1e3:.0f}k int8 embedding + int4 head; KV cache at full context "
              f"{info['kv_cache_bytes'] / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
