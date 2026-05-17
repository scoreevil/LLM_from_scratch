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
        CUDA bf16 autocast by default (``--no-bf16`` to disable); no GradScaler.
        Every --eval-every steps: flip to eval(), score N fixed val batches,
        log {step, train_loss, val_loss, ppl} to JSONL, flip back to train().
        Checkpoints (``--train`` only) go to training/checkpoints/pretrain/:
        last.pt on every eval, best.pt when val_loss improves.

    ``--epochs`` (default 3) replays the full train JSONL that many times.
    ``--steps 0`` means no global step cap; set ``--steps N`` to stop early.
    Before training, batches/epoch is estimated via a fast char scan (default)
    so training can start immediately; use ``--count-batches-exact`` for a full
    tokenize pass or ``--batches-per-epoch N`` to pin the denominator.

Logging:
    train_log.txt is JSONL — one row per eval point. Easy to read with
    pandas.read_json(path, lines=True) for plotting later.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

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

# Real pretrain checkpoints (model + optimizer + hyperparams for resume).
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "pretrain"


def _resolve_tokenizer_dir(tokenizer: str | None, tokenizer_size: str) -> Path:
    if tokenizer:
        return Path(tokenizer)
    return Path("tokenizer") / tokenizer_size


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


def resolve_bf16_amp(device: torch.device, *, enabled: bool) -> Optional[torch.dtype]:
    """Return ``torch.bfloat16`` for CUDA autocast when supported, else ``None``."""
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


def compute_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Standard next-token CE: flatten over batch and time."""
    with amp_autocast(x.device, amp_dtype):
        logits, _ = model(x)                           # [B, T, V]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    amp_dtype: Optional[torch.dtype] = None,
) -> float:
    """Mean CE loss over a fixed list of pre-loaded batches. Toggles eval()/train()."""
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    for x, y in batches:
        loss = compute_loss(model, x, y, amp_dtype=amp_dtype)
        total += loss.item()
        n += 1
    if was_training:
        model.train()
    return total / max(1, n)


def build_train_loader(
    files: List[Path],
    tokenizer,
    seq_len: int,
    batch_size: int,
) -> DataLoader:
    ds = PackedJSONLDataset(files, tokenizer=tokenizer, seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


# Measured chars/token on mix JSONL (tokenizer/eval_token_ratio.py backtest).
_CHARS_PER_TOKEN_DEFAULT = {"zh": 1.254, "en": 3.559}


def packed_batches_from_stream(
    total_tokens: int,
    num_docs: int,
    seq_len: int,
    batch_size: int,
) -> int:
    """Match ``PackedJSONLDataset`` packing: one SEP per doc, then fixed blocks."""
    block = seq_len + 1
    stream = total_tokens + num_docs
    n_seqs = stream // block
    return max(1, (n_seqs + batch_size - 1) // batch_size)


def scan_jsonl_chars(files: List[Path]) -> Dict[str, Dict[str, int]]:
    """One UTF-8 pass per file: per-language char and doc totals (no tokenize)."""
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"chars": 0, "docs": 0})
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                lang = rec.get("lang")
                text = rec.get("text") or ""
                if not lang or not text:
                    continue
                totals[lang]["chars"] += len(text)
                totals[lang]["docs"] += 1
    return dict(totals)


def sample_chars_per_token(
    files: List[Path],
    tokenizer,
    per_lang: int,
) -> Dict[str, float]:
    """Encode up to ``per_lang`` docs per language; return chars/token ratios."""
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"chars": 0, "tokens": 0, "docs": 0})
    target_langs = tuple(_CHARS_PER_TOKEN_DEFAULT)
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if all(stats[lg]["docs"] >= per_lang for lg in target_langs):
                    return {
                        lg: stats[lg]["chars"] / stats[lg]["tokens"]
                        for lg in target_langs
                        if stats[lg]["tokens"] > 0
                    }
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                lang = rec.get("lang")
                text = rec.get("text") or ""
                if lang not in target_langs or stats[lang]["docs"] >= per_lang or not text:
                    continue
                ids = tokenizer.encode(text)
                stats[lang]["chars"] += len(text)
                stats[lang]["tokens"] += len(ids)
                stats[lang]["docs"] += 1
    return {
        lg: stats[lg]["chars"] / stats[lg]["tokens"]
        for lg in target_langs
        if stats[lg]["tokens"] > 0
    }


def estimate_batches_per_epoch(
    files: List[Path],
    tokenizer,
    seq_len: int,
    batch_size: int,
    *,
    chars_per_token: Optional[Dict[str, float]] = None,
    samples_per_lang: int = 0,
) -> Tuple[int, float]:
    """Fast batches/epoch estimate: char scan (+ optional small encode sample).

    Returns ``(batches_per_epoch, estimated_total_tokens)``.
    """
    cpt = dict(_CHARS_PER_TOKEN_DEFAULT)
    if chars_per_token:
        cpt.update(chars_per_token)
    if samples_per_lang > 0:
        t0 = time.time()
        sampled = sample_chars_per_token(files, tokenizer, samples_per_lang)
        cpt.update(sampled)
        print(
            f"[train]   sample encode {samples_per_lang}/lang in {time.time() - t0:.1f}s  "
            + "  ".join(f"{lg}={cpt[lg]:.3f} c/t" for lg in sorted(sampled)),
            flush=True,
        )

    t0 = time.time()
    char_stats = scan_jsonl_chars(files)
    print(f"[train]   char scan {time.time() - t0:.1f}s", flush=True)

    est_tokens = 0.0
    num_docs = 0
    for lang, row in char_stats.items():
        num_docs += row["docs"]
        ratio = cpt.get(lang)
        if ratio and row["chars"]:
            est_tokens += row["chars"] / ratio

    batches = packed_batches_from_stream(
        int(est_tokens), num_docs, seq_len, batch_size
    )
    return batches, est_tokens


def count_batches_per_epoch(
    files: List[Path],
    tokenizer,
    seq_len: int,
    batch_size: int,
) -> int:
    """One full tokenize+pack pass to count training batches in a single epoch."""
    loader = build_train_loader(files, tokenizer, seq_len, batch_size)
    n = 0
    for _ in tqdm(loader, desc="count-batches/epoch", unit="batch"):
        n += 1
    return max(1, n)


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


def save_checkpoint(
    path: Path,
    *,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    train_loss: float,
    model_config: dict,
) -> None:
    """Write a resumable checkpoint (weights, optimizer, step, model_config)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "train_loss": train_loss,
            "model_config": model_config,
        },
        path,
    )


def _epoch_data_pct(epoch_batches: int, batches_per_epoch: int) -> float:
    return min(100.0, 100.0 * epoch_batches / max(1, batches_per_epoch))


def _progress_suffix(epoch: int, epochs: int, epoch_batches: int, batches_per_epoch: int) -> str:
    pct = _epoch_data_pct(epoch_batches, batches_per_epoch)
    return (
        f"epoch {epoch}/{epochs}  "
        f"data {epoch_batches}/{batches_per_epoch} ({pct:.1f}%)"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_loop(
    model: nn.Module,
    new_train_iter: Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    batches_per_epoch: int,
    max_steps: int,
    eval_every: int,
    grad_clip: float,
    log_path: Path,
    device: torch.device,
    print_every: int = 10,
    model_config: Optional[dict] = None,
    ckpt_dir: Optional[Path] = None,
    amp_dtype: Optional[torch.dtype] = None,
) -> None:
    """Train for ``epochs`` full passes; optional global cap ``max_steps`` (0 = none)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else CKPT_DIR
    save_ckpt = model_config is not None
    best_val_loss = float("inf")
    unlimited_steps = max_steps <= 0

    model.train()
    step = 0
    started = time.time()
    last_train_loss = float("nan")
    stop_training = False

    try:
        for epoch in range(1, epochs + 1):
            if stop_training:
                break
            epoch_batches = 0
            train_iter = new_train_iter()
            print(
                f"\n[epoch {epoch}/{epochs}] start  "
                f"({batches_per_epoch} batches/epoch)",
                flush=True,
            )

            while True:
                if not unlimited_steps and step >= max_steps:
                    print(f"[info] reached --steps {max_steps}", flush=True)
                    stop_training = True
                    break

                try:
                    x, y = next(train_iter)
                except StopIteration:
                    pct = _epoch_data_pct(epoch_batches, batches_per_epoch)
                    print(
                        f"[epoch {epoch}/{epochs}] complete  "
                        f"data {epoch_batches}/{batches_per_epoch} ({pct:.1f}%)",
                        flush=True,
                    )
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                loss = compute_loss(model, x, y, amp_dtype=amp_dtype)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                last_train_loss = loss.item()
                step += 1
                epoch_batches += 1
                prog = _progress_suffix(epoch, epochs, epoch_batches, batches_per_epoch)

                if step % print_every == 0:
                    tok_per_s = step * x.size(0) * x.size(1) / max(1e-9, time.time() - started)
                    print(
                        f"step {step:>6d}  {prog}  "
                        f"train_loss={last_train_loss:.4f}  tok/s={tok_per_s:,.0f}",
                        flush=True,
                    )

                do_eval = step % eval_every == 0
                if do_eval:
                    vloss = eval_loss(model, val_batches, amp_dtype=amp_dtype)
                    ppl = math.exp(min(vloss, 50.0))
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "epochs": epochs,
                        "epoch_batches": epoch_batches,
                        "batches_per_epoch": batches_per_epoch,
                        "epoch_data_pct": round(
                            _epoch_data_pct(epoch_batches, batches_per_epoch), 2
                        ),
                        "train_loss": last_train_loss,
                        "val_loss": vloss,
                        "ppl": ppl,
                        "wallclock_s": round(time.time() - started, 2),
                    }
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()
                    print(
                        f"  >> eval @ step {step}  {prog}  "
                        f"val_loss={vloss:.4f}  PPL={ppl:.3f}",
                        flush=True,
                    )

                    if save_ckpt:
                        save_checkpoint(
                            ckpt_dir / "last.pt",
                            step=step,
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            val_loss=vloss,
                            train_loss=last_train_loss,
                            model_config=model_config,
                        )
                        if vloss < best_val_loss:
                            best_val_loss = vloss
                            save_checkpoint(
                                ckpt_dir / "best.pt",
                                step=step,
                                epoch=epoch,
                                model=model,
                                optimizer=optimizer,
                                val_loss=vloss,
                                train_loss=last_train_loss,
                                model_config=model_config,
                            )
                            print(
                                f"  >> saved best.pt (val_loss={vloss:.4f}) -> {ckpt_dir}",
                                flush=True,
                            )
                        else:
                            print(f"  >> saved last.pt -> {ckpt_dir}", flush=True)

            if epoch_batches < batches_per_epoch:
                print(
                    f"[warn] epoch {epoch} ended early: "
                    f"{epoch_batches} < {batches_per_epoch} batches",
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
    tokenizer_dir = _resolve_tokenizer_dir(args.tokenizer, args.tokenizer_size)
    print(f"[train] loading tokenizer from {tokenizer_dir}", flush=True)
    tok = BBPETokenizer.load(tokenizer_dir)
    vocab_size = tok.vocab_size
    print(f"[train] vocab_size = {vocab_size}", flush=True)

    print(f"[train] building model: d_model={args.d_model}, n_layer={args.n_layer}, "
          f"n_heads={args.n_heads}, seq_len={args.seq_len}", flush=True)
    print(f"[train] flash_attn={args.flash_attn}  checkpointing={args.checkpointing}",
          flush=True)
    model = MiniLLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_heads=args.n_heads,
        max_seq_len=args.seq_len,
        ffn_mult=4,
        bias=True,
        tie_weights=True,
        use_flash_attn=args.flash_attn,
        use_checkpointing=args.checkpointing,
    ).to(device)
    n_params = sum({id(p): p.numel() for p in model.parameters()}.values())
    print(f"[train] params = {n_params / 1e6:.2f} M", flush=True)

    amp_dtype = resolve_bf16_amp(device, enabled=not args.no_bf16)
    if amp_dtype is not None:
        print("[train] mixed precision: bfloat16 autocast", flush=True)
    else:
        print("[train] mixed precision: off (fp32)", flush=True)

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

    train_files = [Path(f) for f in args.train_files]
    cpt_override: Dict[str, float] = {}
    if args.chars_per_tok_zh is not None:
        cpt_override["zh"] = args.chars_per_tok_zh
    if args.chars_per_tok_en is not None:
        cpt_override["en"] = args.chars_per_tok_en

    if args.batches_per_epoch > 0:
        batches_per_epoch = args.batches_per_epoch
        est_tokens = batches_per_epoch * args.batch_size * args.seq_len
        print(
            f"[train] using --batches-per-epoch {batches_per_epoch:,}  "
            f"(~{est_tokens/1e6:.1f}M training tokens/epoch, nominal)",
            flush=True,
        )
    elif args.count_batches_exact:
        print("[train] counting batches/epoch (full tokenize pass)...", flush=True)
        batches_per_epoch = count_batches_per_epoch(
            train_files, tok, args.seq_len, args.batch_size
        )
        est_tokens = batches_per_epoch * args.batch_size * args.seq_len
        print(
            f"[train] {batches_per_epoch:,} batches/epoch (exact)  "
            f"(~{est_tokens/1e6:.1f}M tokens/epoch, nominal)",
            flush=True,
        )
    else:
        print("[train] estimating batches/epoch (char scan, no full tokenize)...", flush=True)
        batches_per_epoch, est_tokens = estimate_batches_per_epoch(
            train_files,
            tok,
            args.seq_len,
            args.batch_size,
            chars_per_token=cpt_override or None,
            samples_per_lang=args.estimate_samples_per_lang,
        )
        print(
            f"[train] ~{batches_per_epoch:,} batches/epoch  "
            f"(~{est_tokens/1e6:.1f}M tokens in corpus, est.)",
            flush=True,
        )
        print(
            "[train] progress %% uses this estimate; "
            "pass --count-batches-exact or --batches-per-epoch to pin it.",
            flush=True,
        )
    if args.steps > 0:
        print(f"[train] global step cap: {args.steps:,}", flush=True)
    else:
        print(f"[train] epochs={args.epochs} (no --steps cap)", flush=True)

    def new_train_iter() -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        return iter(build_train_loader(train_files, tok, args.seq_len, args.batch_size))

    log_path = Path(args.log_path)
    ckpt_dir = Path(args.ckpt_dir)
    print(f"[train] log -> {log_path}", flush=True)
    print(f"[train] checkpoints -> {ckpt_dir}", flush=True)

    model_config = {
        "vocab_size": vocab_size,
        "d_model": args.d_model,
        "n_layer": args.n_layer,
        "n_heads": args.n_heads,
        "max_seq_len": args.seq_len,
        "ffn_mult": 4,
        "bias": True,
        "tie_weights": True,
        "use_flash_attn": args.flash_attn,
        "use_checkpointing": args.checkpointing,
    }

    train_loop(
        model=model,
        new_train_iter=new_train_iter,
        val_batches=val_batches,
        optimizer=optimizer,
        epochs=args.epochs,
        batches_per_epoch=batches_per_epoch,
        max_steps=args.steps,
        eval_every=args.eval_every,
        grad_clip=args.clip,
        log_path=log_path,
        device=device,
        print_every=args.print_every,
        model_config=model_config,
        ckpt_dir=ckpt_dir,
        amp_dtype=amp_dtype,
    )

    print(f"[train] done. last.pt / best.pt under {ckpt_dir}", flush=True)


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
    ap.add_argument(
        "--batches-per-epoch",
        type=int,
        default=0,
        help="Pin epoch length for progress %% (0 = estimate or exact count).",
    )
    ap.add_argument(
        "--count-batches-exact",
        action="store_true",
        help="Full tokenize pass to count batches/epoch (slow; default is estimate).",
    )
    ap.add_argument(
        "--estimate-samples-per-lang",
        type=int,
        default=0,
        help="Docs per language to encode when estimating (0 = default c/t only).",
    )
    ap.add_argument(
        "--chars-per-tok-zh",
        type=float,
        default=None,
        help="Override zh chars/token for batch estimate (default 1.254).",
    )
    ap.add_argument(
        "--chars-per-tok-en",
        type=float,
        default=None,
        help="Override en chars/token for batch estimate (default 3.559).",
    )
    # Model
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument(
        "--flash-attn",
        action="store_true",
        help="Use F.scaled_dot_product_attention (FlashAttention on CUDA).",
    )
    ap.add_argument(
        "--checkpointing",
        action="store_true",
        help="Wrap each block in torch.utils.checkpoint (saves activation memory, "
             "trades ~33%% compute). Auto-disabled in eval/generation.",
    )
    # Optim
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16 autocast on CUDA (default: bf16 when supported).",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Full passes over --train-files (default 3).",
    )
    ap.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Global optimizer-step cap (0 = train all epochs).",
    )
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    # Logging
    ap.add_argument("--log-path", default="training/train_log.txt")
    ap.add_argument(
        "--ckpt-dir",
        default=str(CKPT_DIR),
        help="Directory for last.pt / best.pt (use a unique path per experiment).",
    )
    args = ap.parse_args()

    if args.train:
        _pretrain(args)
    else:
        _smoke()


if __name__ == "__main__":
    main()
