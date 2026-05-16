"""Position-wise feed-forward network (Phase 3).

Classic transformer FFN: ``Linear(d -> hidden) -> GELU -> Linear(hidden -> d)``,
default ``hidden = 4 * d_model`` (GPT-2 / Llama-1 convention).

The down-projection is the "residual projection" — i.e. the linear that writes
back into the residual stream — and is tagged ``_is_residual_proj = True`` so
``init_weights(n_layer)`` applies the ``1/sqrt(2N)`` variance scaling.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

try:
    from .layers import GELU
except ImportError:
    # Allow running this file directly as a script.
    from layers import GELU  # type: ignore


class FFN(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, bias: bool = True) -> None:
        super().__init__()
        hidden = d_model * hidden_mult
        self.up = nn.Linear(d_model, hidden, bias=bias)
        self.act = GELU()
        self.down = nn.Linear(hidden, d_model, bias=bias)
        # Mark for scaled init in init_weights(n_layer).
        self.down._is_residual_proj = True

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.act(self.up(x)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke() -> None:
    import math
    import torch

    try:
        from .layers import init_weights
    except ImportError:
        from layers import init_weights  # type: ignore

    torch.manual_seed(0)
    B, T, D = 2, 8, 64

    ffn = FFN(D)
    x = torch.randn(B, T, D)
    y = ffn(x)
    assert y.shape == x.shape
    print(f"  FFN forward    : in {tuple(x.shape)} -> out {tuple(y.shape)}")
    print(f"  hidden dim     : {ffn.up.out_features}  (expect {4 * D})")
    print(f"  down marker    : _is_residual_proj = {ffn.down._is_residual_proj}")
    assert ffn.up.out_features == 4 * D
    assert ffn.down._is_residual_proj is True

    # Verify init_weights respects the marker.
    n_layer = 12
    ffn.apply(init_weights(n_layer=n_layer))
    up_std = ffn.up.weight.std().item()
    down_std = ffn.down.weight.std().item()
    expected_down = 0.02 / math.sqrt(2 * n_layer)
    print(f"  up.weight std  : {up_std:.4f}   (expect ~0.0200)")
    print(f"  down.weight std: {down_std:.4f}   (expect ~{expected_down:.4f})")
    assert abs(up_std - 0.02) < 0.005
    assert abs(down_std - expected_down) < 0.002

    # Device migration.
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for dev in devices:
        ffn_d = FFN(D).to(dev)
        x_d = torch.randn(B, T, D, device=dev)
        y_d = ffn_d(x_d)
        assert y_d.device.type == torch.device(dev).type
        print(f"  [{dev}] FFN forward OK")

    print("\n[FFN smoke OK]")


if __name__ == "__main__":
    _smoke()
