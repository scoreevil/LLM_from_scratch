"""Hand-written core neural primitives (Phase 2).

Provides:
    GELU                - tanh-approximated GELU activation
    LayerNorm           - manual LayerNorm with learnable gamma/beta and epsilon
    RotaryEmbedding     - LLaMA-style RoPE with cached cos/sin buffers
    apply_rope, rotate_half - free functions that apply RoPE to a tensor
    init_weights        - GPT-style weight-init factory (1/sqrt(2N) for residual projections)

Everything is device-agnostic: parameters are nn.Parameter, RoPE caches are
non-persistent buffers, both ride along with module.to(device).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
class GELU(nn.Module):
    """GELU using the tanh approximation (GPT-2 / BERT classic formula).

        GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

    Equivalent to ``torch.nn.GELU(approximate="tanh")`` but written by hand
    so we don't rely on the high-level wrapper.
    """

    # sqrt(2/pi) precomputed as a Python float; cast to tensor in forward
    # to inherit dtype/device from x.
    _SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

    def forward(self, x: Tensor) -> Tensor:
        c = self._SQRT_2_OVER_PI
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))


# ---------------------------------------------------------------------------
# LayerNorm
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm over the last dim, with learnable scale (gamma) and bias (beta).

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    ``var`` is computed with unbiased=False to match the conventional
    population-variance LayerNorm (matches ``torch.nn.LayerNorm``).
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_normed = (x - mean) * torch.rsqrt(var + self.eps)
        return self.gamma * x_normed + self.beta


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (LLaMA style)
# ---------------------------------------------------------------------------
def rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dim by swapping its two halves with a sign flip.

    Used by ``apply_rope``: for a vector ``[x1, x2]`` (each half-length D/2),
    returns ``[-x2, x1]``. Together with the cos/sin terms this implements a
    2D rotation on each (i, i+D/2) pair.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to a tensor whose last two dims are (seq_len, head_dim).

    Args:
        x:   tensor of shape ``[..., T, D]`` (D must be even).
        cos: shape ``[T, D/2]`` (will be duplicated along D to ``[T, D]``).
        sin: shape ``[T, D/2]``.

    Returns:
        Tensor of the same shape as ``x``.
    """
    # Duplicate so each element of a (i, i+D/2) pair sees the same angle.
    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = torch.cat((sin, sin), dim=-1)
    # cos_full / sin_full are [T, D]; broadcasting handles any leading dims
    # of x (batch, heads, ...).
    return x * cos_full + rotate_half(x) * sin_full


class RotaryEmbedding(nn.Module):
    """Precomputes and caches cos/sin tables for RoPE.

    Tables live as non-persistent buffers, so ``.to(device)`` migrates them
    automatically and they do NOT bloat the saved checkpoint.

    Typical use inside attention:
        rope = RotaryEmbedding(head_dim, max_seq_len=2048)
        cos, sin = rope(seq_len=T)              # [T, D/2] each
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE head dim must be even"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [T, D/2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Return (cos, sin) slices of shape ``[seq_len, D/2]``.

        ``offset`` lets the caller fetch a window starting later in the
        sequence — needed once KV caching arrives in a later phase.
        """
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"requested seq_len+offset={end} exceeds max_seq_len={self.max_seq_len}"
            )
        return self.cos_cached[offset:end], self.sin_cached[offset:end]


# ---------------------------------------------------------------------------
# GPT-style init
# ---------------------------------------------------------------------------
def init_weights(n_layer: int, std: float = 0.02) -> Callable[[nn.Module], None]:
    """Return an init function suitable for ``model.apply(...)``.

    Behaviour:
        - ``nn.Linear`` and ``nn.Embedding``: normal(0, std).
        - ``nn.Linear`` flagged with ``_is_residual_proj=True``:
            normal(0, std / sqrt(2 * n_layer)).  This is the GPT-2 trick that
            keeps the residual stream variance from growing with depth.
        - ``nn.Linear`` biases zeroed.
        - Our custom ``LayerNorm`` (gamma=1, beta=0 by construction) is left alone.

    Phase-3 attention/MLP code is responsible for setting
    ``self._is_residual_proj = True`` on the output projection of each
    block — that flag is the only signal this function uses.
    """
    assert n_layer >= 1, "n_layer must be >= 1"
    res_std = std / math.sqrt(2 * n_layer)

    def _fn(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            this_std = res_std if getattr(module, "_is_residual_proj", False) else std
            nn.init.normal_(module.weight, mean=0.0, std=this_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    return _fn


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke() -> None:
    torch.manual_seed(0)

    print("--- shapes & forward pass ---")
    B, T, D = 2, 16, 64

    # GELU
    gelu = GELU()
    x = torch.randn(B, T, D)
    y = gelu(x)
    assert y.shape == x.shape
    print(f"  GELU         : in {tuple(x.shape)} -> out {tuple(y.shape)}")

    # LayerNorm
    ln = LayerNorm(D)
    y = ln(x)
    assert y.shape == x.shape
    # Sanity: per-row mean ~ 0, var ~ 1 (before gamma/beta which are init'd to 1/0).
    m = y.mean(dim=-1).abs().max().item()
    v = y.var(dim=-1, unbiased=False).mean().item()
    assert m < 1e-5, f"LayerNorm mean not ~0: {m}"
    assert abs(v - 1.0) < 1e-4, f"LayerNorm var not ~1: {v}"
    print(f"  LayerNorm    : in {tuple(x.shape)} -> out {tuple(y.shape)}  (mean={m:.2e}, var={v:.4f})")

    # RoPE
    head_dim = 32
    n_heads = 4
    rope = RotaryEmbedding(head_dim, max_seq_len=128)
    cos, sin = rope(seq_len=T)
    q = torch.randn(B, n_heads, T, head_dim)
    k = torch.randn(B, n_heads, T, head_dim)
    q_r = apply_rope(q, cos, sin)
    k_r = apply_rope(k, cos, sin)
    assert q_r.shape == q.shape and k_r.shape == k.shape
    # Norm preservation: rotation must not change the L2 norm of each token vector.
    q_norm_before = q.pow(2).sum(dim=-1).sqrt()
    q_norm_after = q_r.pow(2).sum(dim=-1).sqrt()
    norm_err = (q_norm_before - q_norm_after).abs().max().item()
    assert norm_err < 1e-4, f"RoPE not norm-preserving: {norm_err}"
    print(f"  RoPE         : q {tuple(q.shape)} -> {tuple(q_r.shape)}  "
          f"cos/sin {tuple(cos.shape)}  norm_err={norm_err:.2e}")

    # init_weights
    print("\n--- init_weights ---")
    n_layer = 12
    fake = nn.Sequential(
        nn.Embedding(1000, D),
        nn.Linear(D, 4 * D),     # an MLP up-projection (non-residual)
        nn.Linear(4 * D, D),     # the MLP down-projection -> residual
    )
    fake[2]._is_residual_proj = True  # mark the residual projection
    fake.apply(init_weights(n_layer=n_layer))

    emb_std = fake[0].weight.std().item()
    up_std = fake[1].weight.std().item()
    down_std = fake[2].weight.std().item()
    expected_down = 0.02 / math.sqrt(2 * n_layer)
    print(f"  embedding std  : {emb_std:.4f}   (expect ~0.0200)")
    print(f"  up-proj std    : {up_std:.4f}   (expect ~0.0200)")
    print(f"  down-proj std  : {down_std:.4f}   (expect ~{expected_down:.4f})")
    # Allow generous tolerance: std is a sample statistic.
    assert abs(emb_std - 0.02) < 0.005
    assert abs(up_std - 0.02) < 0.005
    assert abs(down_std - expected_down) < 0.002

    # Device-agnostic check.
    print("\n--- device migration ---")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("  (cuda not available; CPU-only smoke)")

    for dev in devices:
        gelu_d = GELU().to(dev)
        ln_d = LayerNorm(D).to(dev)
        rope_d = RotaryEmbedding(head_dim, max_seq_len=128).to(dev)
        # Verify buffer migration: RoPE's cached tables must follow .to().
        assert rope_d.cos_cached.device.type == torch.device(dev).type, \
            f"RoPE cos_cached did not migrate to {dev}"
        x_d = torch.randn(B, T, D, device=dev)
        q_d = torch.randn(B, n_heads, T, head_dim, device=dev)
        cos_d, sin_d = rope_d(seq_len=T)
        _ = gelu_d(x_d)
        _ = ln_d(x_d)
        _ = apply_rope(q_d, cos_d, sin_d)
        print(f"  [{dev}] GELU / LayerNorm / RoPE forward OK  "
              f"(rope.cos_cached on {rope_d.cos_cached.device})")

    print("\n[Phase 2 smoke OK]")


if __name__ == "__main__":
    _smoke()
