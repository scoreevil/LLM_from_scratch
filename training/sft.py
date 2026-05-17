"""Phase 7 - Supervised Fine-Tuning (SFT) loop.

Loads a Phase-5 pretraining checkpoint, fine-tunes it on ChatML-format
instruction data, and writes the result to
``training/checkpoints/<base>/SFT_<base>.pt``.

Critical design point — Loss Masking
------------------------------------
For each conversation we build a flat token sequence:

    system: ...\\nuser: ...\\nassistant: <RESPONSE>\\n

The ``labels`` tensor is set to ``IGNORE_INDEX`` (-100) for every position
that is NOT inside an assistant response. Only assistant content tokens
contribute to the cross-entropy loss. PyTorch's ``F.cross_entropy`` skips
``-100`` positions automatically.

We deliberately do *not* introduce ChatML special markers like
``<|im_start|>`` — the BBPE vocab doesn't contain them, so they would be
split into arbitrary byte tokens the base model has never seen. Plain
``role: text\\n`` lines keep the prompt fully in-distribution.

Training loop reuses the Phase-5 device/AMP/clip plumbing. Logging is JSONL
to ``--log-path``; checkpoints land in ``--ckpt-dir`` with ``best.pt`` /
``last.pt`` plus a copy at ``<ckpt-dir>/SFT_<base>.pt`` as the canonical
"SFT result" file at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.generate import load_checkpoint  # noqa: E402
from tokenizer import BBPETokenizer  # noqa: E402

IGNORE_INDEX = -100
PAD_ID = 0  # NULL byte token from BBPE base vocab; never appears in real text.

ROLE_PREFIX = {"system": "system: ", "user": "user: ", "assistant": "assistant: "}
TURN_SEP = "\n"


def _resolve_tokenizer_dir(tokenizer: str | None, tokenizer_size: str) -> Path:
    if tokenizer:
        return Path(tokenizer)
    return Path("tokenizer") / tokenizer_size


# ---------------------------------------------------------------------------
# Tokenisation with role-aware label masking
# ---------------------------------------------------------------------------
def encode_chatml(
    messages: List[dict],
    tok: BBPETokenizer,
    max_len: int,
) -> Tuple[List[int], List[int]]:
    """Convert a list of {role, content} into (input_ids, labels).

    For every assistant turn we emit:
        prefix tokens  : "assistant: "             label = IGNORE_INDEX
        content tokens : <message body> + "\\n"    label = real id

    For system / user turns the entire span (prefix + content + sep) is
    masked. Truncation is right-side (drop the tail) so the prompt structure
    is always preserved.
    """
    input_ids: List[int] = []
    labels: List[int] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ROLE_PREFIX:
            continue

        prefix_ids = tok.encode(ROLE_PREFIX[role])
        if role == "assistant":
            content_ids = tok.encode(content + TURN_SEP)
            input_ids.extend(prefix_ids); labels.extend([IGNORE_INDEX] * len(prefix_ids))
            input_ids.extend(content_ids); labels.extend(content_ids)
        else:
            body_ids = tok.encode(content + TURN_SEP)
            input_ids.extend(prefix_ids); labels.extend([IGNORE_INDEX] * len(prefix_ids))
            input_ids.extend(body_ids);   labels.extend([IGNORE_INDEX] * len(body_ids))

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return input_ids, labels


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class ChatMLDataset(Dataset):
    """Loads sft_*.jsonl, pre-tokenises into (input_ids, labels) lists.

    Pre-tokenising up front is cheap (alpaca-scale ~20k items, BBPE is
    fast on small inputs) and gives DataLoader a uniform per-sample cost.
    Skips conversations whose assistant turns are all empty (no positive
    loss positions, would just slow training).
    """

    def __init__(self, path: Path, tok: BBPETokenizer, max_len: int):
        self.items: List[Tuple[List[int], List[int]]] = []
        kept = dropped = 0
        with path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"encode {path.name}", unit="ex"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                msgs = rec.get("messages") or []
                if not msgs:
                    dropped += 1; continue
                ids, lbls = encode_chatml(msgs, tok, max_len)
                if not any(l != IGNORE_INDEX for l in lbls) or len(ids) < 2:
                    dropped += 1; continue
                self.items.append((ids, lbls))
                kept += 1
        print(f"  [data] {path.name}: kept {kept}, dropped {dropped}", flush=True)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Tuple[List[int], List[int]]:
        return self.items[i]


def collate(batch: List[Tuple[List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the batch's longest sequence.

    pad slots: input_ids = PAD_ID, labels = IGNORE_INDEX (so they cost nothing).
    """
    max_len = max(len(b[0]) for b in batch)
    B = len(batch)
    x = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    y = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, lbls) in enumerate(batch):
        L = len(ids)
        x[i, :L] = torch.tensor(ids, dtype=torch.long)
        y[i, :L] = torch.tensor(lbls, dtype=torch.long)
    return x, y


# ---------------------------------------------------------------------------
# AMP helpers (mirror pretrain.py so behaviour matches)
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_bf16_amp(device: torch.device, enabled: bool) -> Optional[torch.dtype]:
    if not enabled or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    print("[warn] --bf16 requested but GPU lacks bf16; training in fp32", flush=True)
    return None


@contextlib.contextmanager
def amp_autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        yield
    else:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            yield


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------
def loss_with_shift(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    amp_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Compute next-token CE with the standard 1-step shift.

    logits[..., :-1, :] predicts y[..., 1:]; the loss is meaningful only on
    label positions != IGNORE_INDEX, which ignore_index handles.
    """
    with amp_autocast(x.device, amp_dtype):
        logits, _ = model(x)                                # [B, T, V]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    max_batches: int,
) -> Tuple[float, float]:
    was_training = model.training
    model.eval()
    losses: List[float] = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss = loss_with_shift(model, x, y, amp_dtype)
        losses.append(loss.item())
    if was_training:
        model.train()
    mean = sum(losses) / max(1, len(losses))
    return mean, math.exp(min(mean, 50.0))


# ---------------------------------------------------------------------------
# Training entry
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> None:
    device = get_device()
    torch.manual_seed(args.seed)
    print(f"[sft] device = {device}", flush=True)

    # --- load base ----------------------------------------------------------
    base_ckpt = Path(args.base_ckpt)
    print(f"[sft] loading base checkpoint: {base_ckpt}", flush=True)
    model, cfg = load_checkpoint(
        base_ckpt,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        device=device,
    )
    n_params = sum({id(p): p.numel() for p in model.parameters()}.values())
    print(f"[sft] base config: vocab={cfg['vocab_size']} d_model={cfg['d_model']} "
          f"n_layer={cfg['n_layer']} n_heads={cfg['n_heads']} "
          f"max_seq_len={cfg['max_seq_len']}  params={n_params/1e6:.2f}M",
          flush=True)

    # SFT effective max len = min(model.max_seq_len, --max-len).
    eff_max = min(int(cfg["max_seq_len"]), args.max_len)
    if eff_max != args.max_len:
        print(f"[sft] capping max_len to model.max_seq_len={cfg['max_seq_len']}",
              flush=True)

    # --- tokenizer + data ---------------------------------------------------
    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer, args.tokenizer_size)
    tok = BBPETokenizer.load(tokenizer_dir)
    print(f"[sft] tokenizer dir = {tokenizer_dir}", flush=True)
    print(f"[sft] tokenizer vocab = {tok.vocab_size}", flush=True)

    train_ds = ChatMLDataset(Path(args.train_file), tok, eff_max)
    val_ds   = ChatMLDataset(Path(args.val_file),   tok, eff_max)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate)

    print(f"[sft] batches/epoch = {len(train_loader)}", flush=True)

    # --- optim --------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    amp_dtype = resolve_bf16_amp(device, enabled=not args.no_bf16)
    print(f"[sft] amp = {amp_dtype}", flush=True)

    # --- output paths -------------------------------------------------------
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")
    # Canonical filename per the spec: SFT_<base>.pt under the group dir.
    base_name = ckpt_dir.name  # e.g. "PT-1A"
    final_path = ckpt_dir / f"SFT_{base_name}.pt"
    best_path  = ckpt_dir / "SFT_best.pt"
    last_path  = ckpt_dir / "SFT_last.pt"
    print(f"[sft] ckpt dir = {ckpt_dir}", flush=True)
    print(f"[sft] final ckpt -> {final_path}", flush=True)

    # --- loop ---------------------------------------------------------------
    model.train()
    step = 0
    best_val = float("inf")
    started = time.time()
    sft_model_config = {**cfg, "sft_from": str(base_ckpt)}

    def save_ckpt(path: Path, *, extra: dict | None = None) -> None:
        payload = {
            "model_state_dict": model.state_dict(),
            "model_config": sft_model_config,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    try:
        for epoch in range(args.epochs):
            print(f"\n[sft] === epoch {epoch + 1}/{args.epochs} ===", flush=True)
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                loss = loss_with_shift(model, x, y, amp_dtype)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
                optimizer.step()
                step += 1

                if step % args.print_every == 0:
                    print(f"  step {step:>6d}  train_loss={loss.item():.4f}  "
                          f"({time.time() - started:.1f}s)", flush=True)

                if step % args.eval_every == 0:
                    val_loss, val_ppl = evaluate(
                        model, val_loader, device, amp_dtype, args.val_batches
                    )
                    row = {
                        "step": step, "epoch": epoch + 1,
                        "train_loss": loss.item(),
                        "val_loss": val_loss, "ppl": val_ppl,
                        "wallclock_s": round(time.time() - started, 2),
                    }
                    log_f.write(json.dumps(row) + "\n"); log_f.flush()
                    flag = ""
                    if val_loss < best_val:
                        best_val = val_loss
                        save_ckpt(best_path, extra={"step": step, "val_loss": val_loss})
                        flag = "  [best]"
                    print(f"  >> eval step {step}: val_loss={val_loss:.4f}  "
                          f"PPL={val_ppl:.3f}{flag}", flush=True)
                    save_ckpt(last_path, extra={"step": step, "val_loss": val_loss})

                if args.steps > 0 and step >= args.steps:
                    print(f"[sft] hit --steps cap ({args.steps}); stopping early.")
                    break
            if args.steps > 0 and step >= args.steps:
                break

        # Final eval and the canonical SFT_<base>.pt artifact.
        val_loss, val_ppl = evaluate(model, val_loader, device, amp_dtype,
                                     max_batches=10_000_000)
        print(f"\n[sft] FINAL val_loss={val_loss:.4f}  PPL={val_ppl:.3f}", flush=True)
        save_ckpt(final_path, extra={"step": step, "val_loss": val_loss,
                                     "final": True})
        print(f"[sft] saved {final_path}", flush=True)
    finally:
        log_f.close()


# ---------------------------------------------------------------------------
# Lightweight prompt-test (loads the SFT result and generates from a prompt)
# ---------------------------------------------------------------------------
def _prompt_test(args: argparse.Namespace) -> None:
    from eval.generate import generate_completion

    device = get_device()
    ckpt = Path(args.sft_ckpt)
    print(f"[test] loading SFT checkpoint: {ckpt}", flush=True)
    model, _ = load_checkpoint(
        ckpt, n_heads=args.n_heads, max_seq_len=args.max_seq_len, device=device
    )
    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer, args.tokenizer_size)
    tok = BBPETokenizer.load(tokenizer_dir)
    print(f"[test] tokenizer dir = {tokenizer_dir}", flush=True)

    prompts = args.test_prompts or [
        "user: 你是谁？\nassistant:",
        "user: 帮我写一段Python代码打印1到10。\nassistant:",
        "user: What's the capital of France?\nassistant:",
    ]
    for p in prompts:
        out = generate_completion(
            model, tok, p,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        print("\n--- prompt ---")
        print(p)
        print("--- continuation ---")
        print(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    # train
    p_t = sub.add_parser("train", help="Run SFT.")
    p_t.add_argument("--base-ckpt", required=True,
                     help="Path to a pretrain checkpoint (.pt or dir w/ best.pt).")
    p_t.add_argument("--train-file", default="data/processed/sft_train.jsonl")
    p_t.add_argument("--val-file",   default="data/processed/sft_val.jsonl")
    p_t.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory path. If omitted, uses tokenizer/<--tokenizer-size>.",
    )
    p_t.add_argument(
        "--tokenizer-size",
        default="8k",
        help="Tokenizer variant under tokenizer/. e.g. 8k, 32k",
    )
    p_t.add_argument("--ckpt-dir",   required=True,
                     help="Where to save SFT outputs. e.g. training/checkpoints/PT-1A")
    p_t.add_argument("--log-path",   default="training/sft_log.txt")
    # model fallbacks
    p_t.add_argument("--n-heads",     type=int, default=4)
    p_t.add_argument("--max-seq-len", type=int, default=512)
    p_t.add_argument("--max-len",     type=int, default=512,
                     help="Cap per-example token length (capped further by model.max_seq_len).")
    # optim
    p_t.add_argument("--epochs",      type=int, default=3)
    p_t.add_argument("--steps",       type=int, default=0,
                     help="Global step cap (0 = train all epochs).")
    p_t.add_argument("--batch-size",  type=int, default=8)
    p_t.add_argument("--lr",          type=float, default=5e-5)
    p_t.add_argument("--weight-decay", type=float, default=0.0)
    p_t.add_argument("--clip",        type=float, default=1.0)
    p_t.add_argument("--no-bf16",     action="store_true")
    p_t.add_argument("--eval-every",  type=int, default=100)
    p_t.add_argument("--val-batches", type=int, default=16)
    p_t.add_argument("--print-every", type=int, default=10)
    p_t.add_argument("--seed",        type=int, default=42)

    # test (load SFT_*.pt and generate)
    p_p = sub.add_parser("test", help="Generate a few completions from an SFT checkpoint.")
    p_p.add_argument("--sft-ckpt",   required=True)
    p_p.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory path. If omitted, uses tokenizer/<--tokenizer-size>.",
    )
    p_p.add_argument(
        "--tokenizer-size",
        default="8k",
        help="Tokenizer variant under tokenizer/. e.g. 8k, 32k",
    )
    p_p.add_argument("--n-heads",    type=int, default=4)
    p_p.add_argument("--max-seq-len", type=int, default=512)
    p_p.add_argument("--max-new-tokens", type=int, default=80)
    p_p.add_argument("--temperature", type=float, default=0.7)
    p_p.add_argument("--top-k",      type=int, default=40)
    p_p.add_argument("--top-p",      type=float, default=0.95)
    p_p.add_argument("--test-prompts", nargs="*", default=None)

    args = ap.parse_args()
    if args.cmd == "train":
        _run(args)
    elif args.cmd == "test":
        _prompt_test(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
