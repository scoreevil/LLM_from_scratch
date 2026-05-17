"""Pre-Norm decoder-only transformer (Phase 4 + 6.6).

Assembles the hand-written pieces from earlier phases:
    layers.py       LayerNorm, RotaryEmbedding, init_weights
    attention.py    CausalMHA  (+ KV-cache, optional Flash kernel)
    ffn.py          FFN

The top-level ``MiniLLM`` is a NanoGPT-shaped decoder LLM with:
    - learned token embedding
    - N stacked Pre-Norm blocks
    - final LayerNorm
    - lm_head (weight-tied with the token embedding by default)
    - GPT-style init applied at construction time

Phase 6.6 additions (memory savers, both default off → fully backward compat):
    use_flash_attn       Switch CausalMHA to F.scaled_dot_product_attention
                         on CUDA. No O(T^2) score tensor. Falls back to the
                         hand-written kernel when False.
    use_checkpointing    Wrap each TransformerBlock's forward in
                         torch.utils.checkpoint.checkpoint, trading ~33% extra
                         compute for substantially smaller activation memory.
                         IMPORTANT: only enabled in train mode (auto-disables
                         when ``self.training`` is False or ``use_cache``
                         is True), so eval/generation is unaffected.

KV-cache contract (inherited from Phase 3):
    past_key_values: Optional[List[KVCache]]
        None   -> no cache, treat input as a fresh prefill.
        list   -> length == n_layer, item i is the (K, V) cache from block i,
                  each of shape [B, H, T_past, head_dim].
    forward(input_ids, past_key_values=None, use_cache=False)
        returns (logits, new_past_key_values), new_past_key_values is None when
        use_cache=False else a freshly-built list of length n_layer.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor

try:
    from .attention import CausalMHA, KVCache
    from .ffn import FFN
    from .layers import LayerNorm, init_weights
except ImportError:
    # Direct-script execution: `python model/transformer.py`.
    from attention import CausalMHA, KVCache  # type: ignore
    from ffn import FFN  # type: ignore
    from layers import LayerNorm, init_weights  # type: ignore


PastKeyValues = Optional[List[KVCache]]


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """Pre-Norm transformer decoder block.

    Forward:
        h = x + attn(ln1(x))
        y = h + ffn(ln2(h))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        ffn_mult: int = 4,
        bias: bool = True,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalMHA(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            bias=bias,
            use_flash_attn=use_flash_attn,
        )
        self.ln2 = LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, hidden_mult=ffn_mult, bias=bias)

    def forward(
        self,
        x: Tensor,
        past_kv: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KVCache]]:
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------
class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layer: int,
        n_heads: int,
        max_seq_len: int = 2048,
        ffn_mult: int = 4,
        bias: bool = True,
        tie_weights: bool = True,
        use_flash_attn: bool = False,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.tie_weights = tie_weights
        self.use_flash_attn = use_flash_attn
        self.use_checkpointing = use_checkpointing

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    ffn_mult=ffn_mult,
                    bias=bias,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(n_layer)
            ]
        )
        self.final_ln = LayerNorm(d_model)
        # lm_head: bias=False is required for weight tying (embedding has no bias).
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # GPT-style init across the whole tree. Must run BEFORE weight tying:
        # otherwise apply() would re-randomize the shared tensor at the lm_head
        # visit and clobber the embedding init.
        self.apply(init_weights(n_layer=n_layer))

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    # ------------------------------------------------------------------ utils
    def num_params(self, exclude_embedding: bool = False) -> int:
        """Total parameter count. Tied weights are counted once."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if exclude_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    # --------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: Tensor,
        past_key_values: PastKeyValues = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, PastKeyValues]:
        """Run the LLM.

        Args:
            input_ids:        LongTensor [B, T].
            past_key_values:  None or a list of length n_layer, each item
                              (K, V) cached from a previous call.
            use_cache:        If True, build and return a new list of KV caches.

        Returns:
            logits:                 FloatTensor [B, T, vocab_size]
            new_past_key_values:    None if use_cache=False, else list of n_layer KVs.
        """
        if past_key_values is not None:
            assert len(past_key_values) == self.n_layer, (
                f"past_key_values must have length n_layer={self.n_layer}, "
                f"got {len(past_key_values)}"
            )

        x = self.tok_emb(input_ids)  # [B, T, D]

        # Gradient checkpointing is only meaningful when we're training and
        # there is no KV cache to grow (cache would be silently discarded
        # by checkpoint's re-forward pass). Disable in eval / generation.
        do_ckpt = (
            self.use_checkpointing
            and self.training
            and not use_cache
            and past_key_values is None
            and torch.is_grad_enabled()
        )

        new_kvs: List[KVCache] = [] if use_cache else None  # type: ignore[assignment]
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if do_ckpt:
                # checkpoint can't easily carry the (out, new_kv) tuple under
                # non-tensor return types in older torch, but it's safe here:
                # we only enable it when use_cache=False, so we just need the
                # tensor out. Wrap in a closure that returns only x.
                def _run(inp, blk=block):
                    out, _ = blk(inp, past_kv=None, use_cache=False)
                    return out
                # use_reentrant=False is the recommended modern path (PT 2.x).
                x = cp.checkpoint(_run, x, use_reentrant=False)
                new_kv = None
            else:
                x, new_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                # new_kv is guaranteed non-None here because use_cache=True.
                new_kvs.append(new_kv)  # type: ignore[arg-type]

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits, new_kvs


# ---------------------------------------------------------------------------
# Smoke test (run by the user — see end of file for the command).
# ---------------------------------------------------------------------------
def _smoke() -> None:
    torch.manual_seed(0)

    cfg = dict(
        vocab_size=8192,
        d_model=256,
        n_layer=12,
        n_heads=8,
        max_seq_len=128,
        ffn_mult=4,
        bias=True,
        tie_weights=True,
    )
    print("--- config ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    model = MiniLLM(**cfg)
    total = model.num_params()
    non_emb = model.num_params(exclude_embedding=True)
    print(f"\n  total params         : {total:,}  ({total / 1e6:.2f} M)")
    print(f"  non-embedding params : {non_emb:,}  ({non_emb / 1e6:.2f} M)")

    # Verify weight tying: lm_head.weight and tok_emb.weight must share storage.
    same_storage = model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    print(f"  weight tying         : {'ON' if same_storage else 'OFF'} "
          f"(lm_head.weight {'shares' if same_storage else 'does NOT share'} tok_emb.weight)")
    assert same_storage, "tie_weights=True but tensors are not shared"

    # 1. Full forward (no cache).
    B, T = 2, 16
    input_ids = torch.randint(0, cfg["vocab_size"], (B, T))
    print(f"\n--- full forward (no cache) ---")
    print(f"  input_ids shape : {tuple(input_ids.shape)}")
    logits, kv = model(input_ids, past_key_values=None, use_cache=False)
    print(f"  logits shape    : {tuple(logits.shape)}  (expect ({B}, {T}, {cfg['vocab_size']}))")
    print(f"  new_past_kv     : {kv}")
    assert logits.shape == (B, T, cfg["vocab_size"])
    assert kv is None

    # 2. Prefill with cache, then a single-token decode step.
    print(f"\n--- KV-cache: prefill then 1-step decode ---")
    pre_logits, past = model(input_ids, use_cache=True)
    print(f"  prefill logits  : {tuple(pre_logits.shape)}")
    print(f"  cache layers    : {len(past)}  (expect {cfg['n_layer']})")
    print(f"  K shape / layer : {tuple(past[0][0].shape)}  "
          f"(expect ({B}, {cfg['n_heads']}, {T}, {cfg['d_model'] // cfg['n_heads']}))")

    next_tok = torch.randint(0, cfg["vocab_size"], (B, 1))
    step_logits, past2 = model(next_tok, past_key_values=past, use_cache=True)
    print(f"  step logits     : {tuple(step_logits.shape)}  (expect ({B}, 1, {cfg['vocab_size']}))")
    print(f"  K shape after   : {tuple(past2[0][0].shape)}  "
          f"(expect ({B}, {cfg['n_heads']}, {T + 1}, {cfg['d_model'] // cfg['n_heads']}))")
    assert step_logits.shape == (B, 1, cfg["vocab_size"])
    assert past2[0][0].shape[-2] == T + 1

    # 3. Device migration (CPU and CUDA if available).
    print(f"\n--- device migration ---")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for dev in devices:
        m = MiniLLM(**cfg).to(dev)
        ids = torch.randint(0, cfg["vocab_size"], (B, T), device=dev)
        out, _ = m(ids)
        assert out.device.type == torch.device(dev).type
        print(f"  [{dev}] forward OK  logits on {out.device}")

    print("\n[Phase 4 smoke OK]")


if __name__ == "__main__":
    _smoke()
