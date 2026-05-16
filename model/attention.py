"""Causal multi-head attention with RoPE and KV-cache (Phase 3).

Conventions:
    - Input layout is ``[B, T, D]`` (batch, time, model_dim).
    - Inside the module q/k/v are reshaped to ``[B, H, T, head_dim]``.
    - ``forward(x, past_kv=None, use_cache=False)`` returns
      ``(out, new_past_kv)``.  ``past_kv`` and ``new_past_kv`` are either
      ``None`` or a ``(K, V)`` tuple of tensors of shape
      ``[B, H, T_past, head_dim]``.
    - RoPE is applied to q and k *after* QKV projection; when ``past_kv``
      is supplied the new positions start at ``offset = T_past`` so the
      single-token decoding step sees the correct phase.
    - The output projection is marked ``_is_residual_proj = True`` so
      ``init_weights(n_layer)`` scales it by ``1/sqrt(2N)``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from .layers import RotaryEmbedding, apply_rope
except ImportError:
    # Direct-script execution.
    from layers import RotaryEmbedding, apply_rope  # type: ignore

KVCache = Tuple[Tensor, Tensor]


class CausalMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Fused QKV projection — one linear, split afterwards.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Output projection (residual writer — tagged for init scaling).
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj._is_residual_proj = True

        # RoPE table for q/k.
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len, base=rope_base)

        # Causal mask buffer — lower-triangular bool of shape [L, L].
        # We slice into this per-call: rows [past_len:past_len+T], cols [:past_len+T].
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        B, T, D = x.shape
        H, Hd = self.n_heads, self.head_dim
        past_len = past_kv[0].shape[-2] if past_kv is not None else 0
        full_len = past_len + T

        if full_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {full_len} exceeds max_seq_len={self.max_seq_len}"
            )

        # --- QKV projection -------------------------------------------------
        qkv = self.qkv(x)                          # [B, T, 3D]
        q, k, v = qkv.split(D, dim=-1)             # each [B, T, D]
        # Reshape to multi-head layout.
        q = q.view(B, T, H, Hd).transpose(1, 2)    # [B, H, T, Hd]
        k = k.view(B, T, H, Hd).transpose(1, 2)
        v = v.view(B, T, H, Hd).transpose(1, 2)

        # --- RoPE on q/k (positions [past_len, past_len + T)) ---------------
        cos, sin = self.rope(seq_len=T, offset=past_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- Append cache ---------------------------------------------------
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)     # [B, H, full_len, Hd]
            v = torch.cat((past_v, v), dim=-2)

        new_past_kv: Optional[KVCache] = (k, v) if use_cache else None

        # --- Scaled dot-product attention -----------------------------------
        # scores: [B, H, T, full_len]
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask: row i (absolute position past_len+i) may attend to
        # keys at columns 0..past_len+i inclusive.
        mask = self.causal_mask[past_len:full_len, :full_len]   # [T, full_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                              # [B, H, T, Hd]

        # Merge heads back.
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)

        return out, new_past_kv


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke() -> None:
    torch.manual_seed(0)

    B, T, D, H = 2, 8, 64, 4
    mha = CausalMHA(d_model=D, n_heads=H, max_seq_len=32)
    x = torch.randn(B, T, D)

    # 1. Forward without cache.
    out_full, kv_none = mha(x, past_kv=None, use_cache=False)
    assert out_full.shape == (B, T, D)
    assert kv_none is None
    print(f"  no-cache fwd    : in {tuple(x.shape)} -> out {tuple(out_full.shape)}  past_kv={kv_none}")

    # 2. Forward with cache (single shot).
    out_cached_full, kv = mha(x, past_kv=None, use_cache=True)
    assert kv is not None
    K, V = kv
    assert K.shape == (B, H, T, D // H)
    assert V.shape == (B, H, T, D // H)
    print(f"  cache fwd       : k.shape={tuple(K.shape)}  v.shape={tuple(V.shape)}")
    # use_cache=True with same input must produce the same output as use_cache=False.
    torch.testing.assert_close(out_full, out_cached_full, rtol=1e-5, atol=1e-5)

    # 3. KV-cache equivalence: prefill T/2 then step one token at a time.
    split = T // 2
    out_pre, kv = mha(x[:, :split, :], past_kv=None, use_cache=True)
    outs = [out_pre]
    for t in range(split, T):
        out_step, kv = mha(x[:, t : t + 1, :], past_kv=kv, use_cache=True)
        outs.append(out_step)
    out_inc = torch.cat(outs, dim=1)
    assert out_inc.shape == out_full.shape
    # Tight tolerance — under fp32 these should match to ~1e-6.
    torch.testing.assert_close(out_full, out_inc, rtol=1e-4, atol=1e-5)
    max_diff = (out_full - out_inc).abs().max().item()
    print(f"  KV-cache equiv  : prefill {split} + step {T - split} == full forward "
          f"(max diff = {max_diff:.2e})")

    # 4. Final cache state should hold T keys/values.
    assert kv[0].shape[-2] == T

    # 5. Out-of-range check: requesting past max_seq_len must raise.
    over = CausalMHA(d_model=D, n_heads=H, max_seq_len=4)
    try:
        over(torch.randn(1, 8, D))
    except ValueError as e:
        print(f"  oob check       : correctly raised ValueError ({e})")
    else:
        raise AssertionError("expected ValueError for seq_len > max_seq_len")

    # 6. Device migration.
    print("\n--- device migration ---")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for dev in devices:
        m = CausalMHA(d_model=D, n_heads=H, max_seq_len=32).to(dev)
        x_d = torch.randn(B, T, D, device=dev)
        # Verify buffers migrated.
        assert m.causal_mask.device.type == torch.device(dev).type
        assert m.rope.cos_cached.device.type == torch.device(dev).type
        # Prefill then one decode step.
        out_p, kv_d = m(x_d[:, :4, :], use_cache=True)
        out_s, kv_d = m(x_d[:, 4:5, :], past_kv=kv_d, use_cache=True)
        assert out_p.shape == (B, 4, D) and out_s.shape == (B, 1, D)
        assert kv_d[0].shape[-2] == 5
        print(f"  [{dev}] prefill + 1-step decode OK  "
              f"(mask on {m.causal_mask.device}, rope on {m.rope.cos_cached.device})")

    print("\n[CausalMHA smoke OK]")


if __name__ == "__main__":
    _smoke()
