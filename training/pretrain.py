"""Phase 5 pretraining loop.

Two CLI modes:
    (default)  Overfit smoke   - tiny in-memory random corpus, no jsonl/tokenizer
                                 needed. Verifies the loop drives val_loss -> ~0
                                 and PPL -> ~1.
    --train    Real pretraining - streams a packed BBPE-tokenized corpus from
                                  data/processed/*.jsonl, evaluates against
                                  data/processed/val_set.jsonl every N steps,
                                  appends each eval row to training/train_log.txt.

Pipeline:
    PackedJSONLDataset
        Stream JSONL -> BBPE encode -> concatenate with SEP_TOKEN (id=0, the
        NULL byte — never appears in UTF-8 text) -> chunk into seq_len+1 slices
        -> yield (input_ids[:-1], input_ids[1:]).

    Loop:
        AdamW(betas=(0.9, 0.95)) + cross_entropy + clip_grad_norm_(1.0).
        Device auto: cuda if available else cpu.
        Every --eval-every steps: flip to eval(), score N fixed val batches,
        log {step, train_loss, val_loss, ppl} to JSONL, flip back to train().

Logging:
    train_log.txt is JSONL — one row per eval point. Easy to read with
    pandas.read_json(path, lines=True) for plotting later.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Allow `from model import ...` / `from tokenizer import ...` when invoked as
# a script from the project root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import MiniLLM  # noqa: E402


# Doc-boundary separator: id 0 = the NULL byte in our byte-level BBPE base
# vocab. NULL never appears in valid UTF-8 text, so it is a clean boundary
# marker without overloading a meaningful token.
SEP_TOKEN_ID = 0


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class PackedJSONLDataset(IterableDataset):
    """Stream tokenized .jsonl docs, pack into fixed-length training sequences.

    Each yielded item is a pair of LongTensors ``(input_ids, labels)`` of
    shape ``[seq_len]``, where ``labels`` is ``input_ids`` shifted left by 1
    (i.e. labels[t] is the token that should come at position t+1).
    Documents are concatenated with ``sep_token_id`` between them.
    """

    def __init__(
        self,
        files: List[Path],
        tokenizer,
        seq_len: int,
        sep_token_id: int = SEP_TOKEN_ID,
        max_docs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.files = [Path(f) for f in files]
        self.tok = tokenizer
        self.seq_len = seq_len
        self.sep = sep_token_id
        self.max_docs = max_docs

    def _iter_docs(self) -> Iterator[str]:
        n = 0
        for path in self.files:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if self.max_docs is not None and n >= self.max_docs:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    text = rec.get("text")
                    if text:
                        yield text
                        n += 1

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        buf: List[int] = []
        block = self.seq_len + 1   # need +1 token to shift for labels
        for text in self._iter_docs():
            ids = self.tok.encode(text)
            buf.extend(ids)
            buf.append(self.sep)
            while len(buf) >= block:
                chunk = buf[:block]
                buf = buf[block:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


class DummyOverfitDataset(Dataset):
    """Fixed random token sequences for the overfit smoke test."""

    def __init__(self, vocab_size: int, n_seqs: int = 4, seq_len: int = 16, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_seqs, seq_len + 1), generator=gen)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data[i]
        return row[:-1].clone(), row[1:].clone()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Standard next-token CE: flatten over batch and time."""
    logits, _ = model(x)                               # [B, T, V]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def eval_loss(model: nn.Module, batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Mean CE loss over a fixed list of pre-loaded batches. Toggles eval()/train()."""
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    for x, y in batches:
        loss = compute_loss(model, x, y)
        total += loss.item()
        n += 1
    if was_training:
        model.train()
    return total / max(1, n)


def prepare_val_batches(
    file: Path,
    tokenizer,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Tokenize a slice of the val set once and pin it to device for repeated eval."""
    ds = PackedJSONLDataset([file], tokenizer=tokenizer, seq_len=seq_len)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    bx: List[torch.Tensor] = []
    by: List[torch.Tensor] = []
    for x, y in ds:
        bx.append(x)
        by.append(y)
        if len(bx) == batch_size:
            batches.append((torch.stack(bx).to(device), torch.stack(by).to(device)))
            bx, by = [], []
            if len(batches) >= n_batches:
                break
    return batches


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_loop(
    model: nn.Module,
    train_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    max_steps: int,
    eval_every: int,
    grad_clip: float,
    log_path: Path,
    device: torch.device,
    print_every: int = 10,
) -> None:
    """Run training for ``max_steps`` steps with periodic eval & JSONL logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    model.train()
    step = 0
    started = time.time()
    last_train_loss = float("nan")

    try:
        while step < max_steps:
            try:
                x, y = next(train_iter)
            except StopIteration:
                print(f"[info] train iterator exhausted at step {step}")
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            loss = compute_loss(model, x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            last_train_loss = loss.item()
            step += 1

            if step % print_every == 0:
                tok_per_s = step * x.size(0) * x.size(1) / max(1e-9, time.time() - started)
                print(
                    f"step {step:>6d}  train_loss={last_train_loss:.4f}  "
                    f"tok/s={tok_per_s:,.0f}",
                    flush=True,
                )

            if step % eval_every == 0 or step == max_steps:
                vloss = eval_loss(model, val_batches)
                ppl = math.exp(min(vloss, 50.0))  # cap to avoid inf for early steps
                row = {
                    "step": step,
                    "train_loss": last_train_loss,
                    "val_loss": vloss,
                    "ppl": ppl,
                    "wallclock_s": round(time.time() - started, 2),
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                print(
                    f"  >> eval @ step {step}: val_loss={vloss:.4f}  PPL={ppl:.3f}",
                    flush=True,
                )
    finally:
        log_f.close()


# ---------------------------------------------------------------------------
# Overfit smoke
# ---------------------------------------------------------------------------
def _smoke() -> None:
    """Memorize a tiny random corpus. Loss should plummet, PPL approach 1."""
    device = get_device()
    torch.manual_seed(0)

    vocab_size = 256
    seq_len = 16
    n_seqs = 4
    batch_size = 2
    n_epochs = 80

    print(f"[smoke] device = {device}")

    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=64,
        n_layer=2,
        n_heads=4,
        max_seq_len=seq_len + 4,
        ffn_mult=4,
        bias=True,
        tie_weights=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model params = {n_params:,}")

    train_ds = DummyOverfitDataset(vocab_size, n_seqs=n_seqs, seq_len=seq_len, seed=42)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_batches = [(x.to(device), y.to(device)) for x, y in val_loader]

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))

    log_path = Path(__file__).resolve().parent / "train_log_smoke.txt"
    if log_path.exists():
        log_path.unlink()
    log_f = log_path.open("w", encoding="utf-8")

    step = 0
    first_train_loss = None
    last_train_loss = None

    for epoch in range(n_epochs):
        epoch_losses: List[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            loss = compute_loss(model, x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
            step += 1
        avg_train = sum(epoch_losses) / len(epoch_losses)
        if first_train_loss is None:
            first_train_loss = avg_train
        last_train_loss = avg_train

        if (epoch + 1) % 5 == 0 or epoch in (0, n_epochs - 1):
            vloss = eval_loss(model, val_batches)
            ppl = math.exp(min(vloss, 50.0))
            row = {
                "step": step,
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "val_loss": vloss,
                "ppl": ppl,
            }
            log_f.write(json.dumps(row) + "\n")
            log_f.flush()
            print(
                f"  epoch {epoch + 1:3d}  step {step:5d}  "
                f"train_loss={avg_train:.4f}  val_loss={vloss:.4f}  PPL={ppl:.3f}",
                flush=True,
            )

    log_f.close()

    final_val = eval_loss(model, val_batches)
    final_ppl = math.exp(min(final_val, 50.0))
    print(
        f"\n[smoke result] first train_loss={first_train_loss:.4f}  "
        f"last train_loss={last_train_loss:.4f}  "
        f"final val_loss={final_val:.4f}  PPL={final_ppl:.3f}"
    )
    print(f"[smoke result] log written to: {log_path}")
    assert final_val < 0.5, (
        f"Overfit smoke FAILED: final val_loss={final_val:.4f}, PPL={final_ppl:.3f}. "
        "Expected loss < 0.5 (PPL < 1.65) when memorizing tiny fixed data."
    )
    print("[overfit smoke OK]  loss collapsed -> 0, PPL collapsed -> 1, full pipeline wired correctly.")


# ---------------------------------------------------------------------------
# Real pretraining entry
# ---------------------------------------------------------------------------
def _pretrain(args: argparse.Namespace) -> None:
    """Full pretraining on BBPE-tokenized JSONL data."""
    device = get_device()
    torch.manual_seed(args.seed)

    # Lazy import — only needed in real-train mode.
    from tokenizer import BBPETokenizer

    print(f"[train] device = {device}", flush=True)
    print(f"[train] loading tokenizer from {args.tokenizer}", flush=True)
    tok = BBPETokenizer.load(args.tokenizer)
    vocab_size = tok.vocab_size
    print(f"[train] vocab_size = {vocab_size}", flush=True)

    print(f"[train] building model: d_model={args.d_model}, n_layer={args.n_layer}, "
          f"n_heads={args.n_heads}, seq_len={args.seq_len}", flush=True)
    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        ffn_mult=4,
        bias=True,
        tie_weights=True,
    ).to(device)
    n_params = sum({id(p): p.numel() for p in model.parameters()}.values())
    print(f"[train] params = {n_params / 1e6:.2f} M", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Prepare val batches once.
    print(f"[train] preparing {args.val_batches} val batches from {args.val_file}", flush=True)
    val_batches = prepare_val_batches(
        file=Path(args.val_file),
        tokenizer=tok,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        n_batches=args.val_batches,
        device=device,
    )
    print(f"[train] got {len(val_batches)} val batches "
          f"({args.batch_size}x{args.seq_len} each)", flush=True)

    # Train iterator (streaming, packed).
    train_ds = PackedJSONLDataset(
        files=[Path(f) for f in args.train_files],
        tokenizer=tok,
        seq_len=args.seq_len,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    train_iter = iter(train_loader)

    log_path = Path(args.log_path)
    print(f"[train] log -> {log_path}", flush=True)

    train_loop(
        model=model,
        train_iter=train_iter,
        val_batches=val_batches,
        optimizer=optimizer,
        max_steps=args.steps,
        eval_every=args.eval_every,
        grad_clip=args.clip,
        log_path=log_path,
        device=device,
        print_every=args.print_every,
    )

    print("[train] done.", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="Run real pretraining. Default (no flag) runs the overfit smoke.")
    # Data
    ap.add_argument("--train-files", nargs="+",
                    default=["data/processed/mix_1to1.jsonl"])
    ap.add_argument("--val-file", default="data/processed/val_set.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer")
    # Model
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    # Optim
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    # Logging
    ap.add_argument("--log-path", default="training/train_log.txt")
    args = ap.parse_args()

    if args.train:
        _pretrain(args)
    else:
        _smoke()


if __name__ == "__main__":
    main()
