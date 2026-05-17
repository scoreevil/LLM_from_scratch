"""Text generation CLI for pretrained checkpoints (Phase 6 helper).

Provides:
    - ``load_checkpoint(...)`` for other evaluators (e.g. benchmarks script)
    - one-shot generation via ``--prompt``
    - interactive REPL when ``--prompt`` is omitted
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import MiniLLM, PastKeyValues  # noqa: E402
from tokenizer import BBPETokenizer  # noqa: E402


def _resolve_tokenizer_dir(tokenizer: str | None, tokenizer_size: str) -> Path:
    """Resolve tokenizer directory.

    Priority:
      1) explicit --tokenizer path
      2) tokenizer/<tokenizer_size> (default: tokenizer/8k)
    """
    if tokenizer:
        return Path(tokenizer)
    return Path("tokenizer") / tokenizer_size


def _safe_torch_load(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_n_layer(state_dict: Dict[str, torch.Tensor]) -> int:
    block_ids = []
    pat = re.compile(r"^blocks\.(\d+)\.")
    for k in state_dict:
        m = pat.match(k)
        if m:
            block_ids.append(int(m.group(1)))
    if not block_ids:
        raise ValueError("cannot infer n_layer from state_dict")
    return max(block_ids) + 1


def _resolve_model_config(
    payload: Dict,
    state_dict: Dict[str, torch.Tensor],
    *,
    n_heads_fallback: int,
    max_seq_len_fallback: int,
) -> Dict:
    cfg = dict(payload.get("model_config") or {})

    tok_w = state_dict.get("tok_emb.weight")
    if tok_w is None:
        raise ValueError("checkpoint missing tok_emb.weight")
    cfg.setdefault("vocab_size", int(tok_w.shape[0]))
    cfg.setdefault("d_model", int(tok_w.shape[1]))
    cfg.setdefault("n_layer", _infer_n_layer(state_dict))

    # n_heads cannot be uniquely recovered from weights when d_model is fixed.
    cfg.setdefault("n_heads", int(n_heads_fallback))

    # Try to recover max_seq_len from RoPE cache if present, otherwise fallback.
    rope_key = "blocks.0.attn.rope.cos_cached"
    if "max_seq_len" not in cfg:
        if rope_key in state_dict and state_dict[rope_key].ndim >= 1:
            cfg["max_seq_len"] = int(state_dict[rope_key].shape[0])
        else:
            cfg["max_seq_len"] = int(max_seq_len_fallback)

    # Infer ffn_mult if absent.
    ffn_w = state_dict.get("blocks.0.ffn.fc_in.weight")
    if "ffn_mult" not in cfg and ffn_w is not None:
        hidden = int(ffn_w.shape[0])
        cfg["ffn_mult"] = max(1, hidden // int(cfg["d_model"]))
    cfg.setdefault("ffn_mult", 4)

    cfg.setdefault("bias", "blocks.0.attn.qkv.bias" in state_dict)
    cfg.setdefault("tie_weights", True)
    return cfg


def load_checkpoint(
    checkpoint: Path | str,
    *,
    n_heads: int = 4,
    max_seq_len: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[MiniLLM, Dict]:
    """Load checkpoint and return ``(model, resolved_config)``."""
    ckpt_path = Path(checkpoint)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = _safe_torch_load(ckpt_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected checkpoint format: {type(payload)}")

    state_dict = payload.get("model_state_dict")
    if state_dict is None:
        # allow raw state_dict checkpoints
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint does not contain a valid state_dict")

    cfg = _resolve_model_config(
        payload,
        state_dict,
        n_heads_fallback=n_heads,
        max_seq_len_fallback=max_seq_len,
    )

    model = MiniLLM(
        vocab_size=int(cfg["vocab_size"]),
        d_model=int(cfg["d_model"]),
        n_layer=int(cfg["n_layer"]),
        n_heads=int(cfg["n_heads"]),
        max_seq_len=int(cfg["max_seq_len"]),
        ffn_mult=int(cfg["ffn_mult"]),
        bias=bool(cfg["bias"]),
        tie_weights=bool(cfg["tie_weights"]),
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"lm_head.weight"}  # tied-weight setups may omit this key
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"missing keys in checkpoint: {bad_missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")

    model.eval()
    return model, cfg


def _sample_next_id(
    logits_1v: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits_1v).item())

    scores = logits_1v / max(1e-6, temperature)

    if top_k > 0 and top_k < scores.numel():
        v, _ = torch.topk(scores, top_k)
        cutoff = v[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(scores, descending=True)
        sorted_probs = F.softmax(sorted_scores, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        to_drop = cdf > top_p
        to_drop[1:] = to_drop[:-1].clone()
        to_drop[0] = False
        sorted_scores = sorted_scores.masked_fill(to_drop, float("-inf"))
        probs = torch.zeros_like(scores)
        probs.scatter_(0, sorted_idx, F.softmax(sorted_scores, dim=-1))
    else:
        probs = F.softmax(scores, dim=-1)

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_completion(
    model: MiniLLM,
    tok: BBPETokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> str:
    input_ids = tok.encode(prompt)
    if not input_ids:
        raise ValueError("prompt encodes to empty token list")

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    logits, past = model(x, use_cache=True)  # prefill
    next_logits = logits[0, -1, :]

    new_ids: List[int] = []
    for _ in range(max_new_tokens):
        next_id = _sample_next_id(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        new_ids.append(next_id)

        # Stop if cache already reached configured length.
        if past is not None and past[0][0].shape[-2] >= model.max_seq_len:
            break

        x_step = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, past = model(x_step, past_key_values=past, use_cache=True)
        next_logits = logits[0, -1, :]

    return tok.decode(new_ids)


def _repl(
    model: MiniLLM,
    tok: BBPETokenizer,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> None:
    print("[REPL] enter prompt and press Enter (Ctrl+C to exit)")
    while True:
        prompt = input("\n>>> ").strip()
        if not prompt:
            continue
        try:
            out = generate_completion(
                model,
                tok,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
            )
            print(prompt + out)
        except Exception as exc:
            print(f"[error] {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory path. If omitted, uses tokenizer/<--tokenizer-size>.",
    )
    ap.add_argument(
        "--tokenizer-size",
        default="8k",
        help="Tokenizer variant under tokenizer/. e.g. 8k, 32k",
    )
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument(
        "--n-heads",
        type=int,
        default=4,
        help="fallback only when checkpoint lacks model_config.n_heads",
    )
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer, args.tokenizer_size)
    tok = BBPETokenizer.load(tokenizer_dir)
    print(f"[load] tokenizer_dir={tokenizer_dir}", flush=True)
    model, cfg = load_checkpoint(
        args.checkpoint,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        device=device,
    )
    print(
        "[load] resolved config: "
        f"vocab={cfg['vocab_size']} d_model={cfg['d_model']} "
        f"n_layer={cfg['n_layer']} n_heads={cfg['n_heads']} "
        f"max_seq_len={cfg['max_seq_len']}",
        flush=True,
    )

    if args.prompt is None:
        _repl(
            model,
            tok,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        return

    out = generate_completion(
        model,
        tok,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(args.prompt + out)


if __name__ == "__main__":
    main()
