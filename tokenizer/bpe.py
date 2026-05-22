"""Hand-written Byte-Level BPE tokenizer (Phase 1).

Pipeline:
    1. UTF-8 bytes -> reversible unicode (the GPT-2 ``bytes_to_unicode`` trick),
       so every byte sequence is round-trippable and merges.txt stays
       human-readable.
    2. Pre-tokenize text with a GPT-4/cl100k style pattern via the ``regex``
       library (needed for ``\\p{L}`` / ``\\p{N}`` and possessive quantifiers).
    3. Count unique pieces; represent each piece as a tuple of single
       byte-unicode chars.
    4. Iteratively merge the most frequent adjacent pair, using an inverted
       ``pair -> set(words)`` index so each merge only re-scans the words
       that actually contain the chosen pair.

Storage:
    vocab.json  - {token_str -> id}
    merges.txt  - "tok_a tok_b" lines in priority order

中文概要：
    本模块实现手写 Byte-Level BPE（BBPE）：先把 UTF-8 字节经 ``bytes_to_unicode``
    映射为可逆的 Unicode 单字符，再按 GPT-4 风格正则切「片段」，在片段内统计
    相邻符号对并迭代合并；用 ``pair -> 包含该对的词集合`` 倒排索引，使每次合并
    只更新受影响的词。词表与合并规则分别存为 ``vocab.json``、``merges.txt``。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import regex as re
from tqdm import tqdm

# Windows 控制台常默认 GBK；字节→Unicode 映射会产生 GBK 无法编码的字符（如 'ł'、'Ġ'），
# 进度条日志可能因此崩溃。这里尽量把 stdout/stderr 设为 UTF-8 并用替换策略，避免跑挂。
for _stream in (sys.stdout, sys.stderr):
    reconfig = getattr(_stream, "reconfigure", None)
    if reconfig is not None:
        try:
            reconfig(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


# GPT-4 / cl100k_base 风格的预分词正则：缩写、字母串、数字、标点与空白等分块。
GPT4_PATTERN = (
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)"""
    r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
    r"""|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


def bytes_to_unicode() -> dict[int, str]:
    """Reversible byte (0-255) -> single-unicode-char mapping (GPT-2 trick).

    Printable ASCII + Latin-1 supplement stay as themselves so merges.txt
    is readable; the remaining 68 bytes (control chars, space, DEL, ...)
    are remapped to chr(256+n). Crucially, the space byte (0x20) becomes
    ``Ġ``, so no token contains a literal space and the "tok_a tok_b"
    separator in merges.txt stays unambiguous.

    中文：将 0–255 每个字节映射到一个 Unicode 字符，保证任意字节序列可逆；
    可打印 ASCII 与部分 Latin-1 保持原样便于阅读；空格字节 0x20 映射为 ``Ġ``，
    这样词表里不会出现字面空格，``merges.txt`` 里用空格分隔两个子词不会歧义。
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# ---------------------------------------------------------------------------
# Multiprocessing workers (must be top-level for Windows `spawn` picklability).
# ---------------------------------------------------------------------------

# 当 `affected` 比这个小，并行分发的 IPC 成本超过节省的计算量,直接走单进程路径。
# 早期 merge 一次性 affected 上万,后期降到几十,该阈值兼顾两端。
APPLY_PARALLEL_THRESHOLD = 256

# Pre-tokenization 分块大小:每个 worker 一次处理这么多 doc,平摊 IPC 与启动成本。
PRETOK_CHUNK_SIZE = 256


def _pretok_chunk(args):
    """Worker:对一批文本应用预分词正则,返回片段->计数的局部 Counter。

    Counter 加法可交换可结合,主进程聚合时与单进程结果完全一致。
    """
    pattern_src, texts = args
    pat = re.compile(pattern_src)
    c: Counter = Counter()
    for t in texts:
        if t:
            c.update(pat.findall(t))
    return c


def _apply_merge_chunk(args):
    """Worker:把单次 merge 应用到一批 `affected` 词上,返回五元组 delta。

    Args:
        best:          被合并的二元组 (a, b)
        new_token:     a + b 的字符串拼接
        chunk_words:   该 worker 负责的词元组列表(`words` dict 的一部分 key)
        chunk_freqs:   {word: freq} dict,仅 chunk_words 子集

    Returns:
        freq_deltas:  dict[pair, int]    某二元组频次的净变化(可正可负)
        add_pw:       dict[pair, list]   某二元组 *新增* 到哪些(新)词中
        rem_pw:       dict[pair, list]   某二元组 *移除* 自哪些(旧)词中
        new_words:    dict[tuple, int]   该 chunk 产生的新词及其频次
        removed:      list[tuple]        该 chunk 处理的旧词(主进程统一从全局删)

    决定论保证:整数加减可交换;集合 add/discard 幂等;
    主进程按 chunk_id 升序应用 → 与单进程顺序处理 word 等价。
    """
    best, new_token, chunk_words, chunk_freqs = args
    a, b = best
    freq_deltas: defaultdict = defaultdict(int)
    add_pw: defaultdict = defaultdict(list)
    rem_pw: defaultdict = defaultdict(list)
    new_words: defaultdict = defaultdict(int)
    removed: list = []
    for old_word in chunk_words:
        freq = chunk_freqs[old_word]
        removed.append(old_word)
        # 从所有「非本次合并对」的统计中扣除该旧词元组的贡献。
        for i in range(len(old_word) - 1):
            p = (old_word[i], old_word[i + 1])
            if p == best:
                continue
            freq_deltas[p] -= freq
            rem_pw[p].append(old_word)
        # 在原词元组上就地应用合并:连续 best 合并为 new_token。
        new_word_list = []
        i = 0
        last = len(old_word) - 1
        while i < len(old_word):
            if i < last and old_word[i] == a and old_word[i + 1] == b:
                new_word_list.append(new_token)
                i += 2
            else:
                new_word_list.append(old_word[i])
                i += 1
        new_word = tuple(new_word_list)
        new_words[new_word] += freq
        # 新词元组产生的相邻对,把频次加回统计与倒排索引。
        for i in range(len(new_word) - 1):
            p = (new_word[i], new_word[i + 1])
            freq_deltas[p] += freq
            add_pw[p].append(new_word)
    return freq_deltas, add_pw, rem_pw, new_words, removed


class BBPETokenizer:
    """Byte-Level BPE 分词器：训练时学 merges 与扩展词表；编码时对每个预分词片段做 BPE 切分。"""

    def __init__(self) -> None:
        self.pat = re.compile(GPT4_PATTERN)
        self.b2u: dict[int, str] = bytes_to_unicode()
        self.u2b: dict[str, int] = {v: k for k, v in self.b2u.items()}
        self.encoder: dict[str, int] = {}  # 子词字符串 -> id
        self.decoder: dict[int, str] = {}  # id -> 子词字符串
        self.bpe_ranks: dict[tuple[str, str], int] = {}  # 合并对 -> 优先级（越小越早合并）
        self._cache: dict[str, list[str]] = {}  # 片段经 BPE 后的切分结果缓存

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # -------------------------------- train ---------------------------------
    def _init_byte_vocab(self) -> None:
        """重置为 256 个单字节子词的基础词表，并清空 BPE 秩与缓存。"""
        self.encoder = {self.b2u[b]: b for b in range(256)}
        self.decoder = {b: self.b2u[b] for b in range(256)}
        self.bpe_ranks = {}
        self._cache.clear()

    def train(
        self,
        iterator: Iterable[str],
        vocab_size: int,
        verbose: bool = True,
        log_every: int = 500,
        num_workers: int = 0,
    ) -> None:
        """在语料上统计片段与相邻对,迭代合并直到词表达到 ``vocab_size``。

        ``num_workers``:
            0 (默认,推荐) -> 单进程,行为与历史版本逐字节一致。
            >=2          -> 多进程并行预分词 + merge-apply 内层。决定论保证产物
                            与单进程逐字节一致(按 chunk_id 升序合并 delta,
                            整数运算可交换,集合操作幂等)。

        ⚠️ 性能现实(Windows):
            spawn 模式下每次 pool.map 都要 pickle 大量 tuple[str] word 对象,
            实测 IPC 成本与计算时间相当甚至更大。num_workers=8 在 Windows 上
            常常**比单进程更慢**。本路径主要价值是 Linux fork 模型(子进程
            COW 共享父内存,IPC 几乎免费),那里能拿到 ~2x 加速。
            **Windows 用户请保持 num_workers=0。**
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (byte-level base)"
        self._init_byte_vocab()

        pool = None
        if num_workers and num_workers >= 2:
            pool = mp.Pool(num_workers)
            if verbose:
                print(f"  [pool] using {num_workers} workers", flush=True)

        try:
            t0 = time.time()
            piece_counts: defaultdict[str, int] = defaultdict(int)
            n_docs = 0

            if pool is None:
                # 单进程预分词 (原路径)。
                pbar = tqdm(
                    iterator,
                    desc="pre-tok",
                    unit="doc",
                    disable=not verbose,
                    mininterval=0.5,
                    dynamic_ncols=True,
                )
                for text in pbar:
                    if not text:
                        continue
                    for piece in self.pat.findall(text):
                        piece_counts[piece] += 1
                    n_docs += 1
                    if n_docs % 2000 == 0:
                        pbar.set_postfix(pieces=len(piece_counts), refresh=False)
                pbar.close()
            else:
                # 多进程预分词:按 PRETOK_CHUNK_SIZE 切批,workers 返回 Counter,
                # 主进程累加。Counter 加法可交换可结合,结果与单进程一致。
                def _chunked():
                    buf: list = []
                    for text in iterator:
                        if not text:
                            continue
                        buf.append(text)
                        if len(buf) >= PRETOK_CHUNK_SIZE:
                            yield (GPT4_PATTERN, buf)
                            buf = []
                    if buf:
                        yield (GPT4_PATTERN, buf)

                pbar = tqdm(
                    desc="pre-tok",
                    unit="chunk",
                    disable=not verbose,
                    mininterval=0.5,
                    dynamic_ncols=True,
                )
                # imap_unordered 拿回每个 chunk 的局部 Counter; 顺序无关。
                for partial in pool.imap_unordered(_pretok_chunk, _chunked(),
                                                   chunksize=1):
                    for piece, c in partial.items():
                        piece_counts[piece] += c
                    n_docs += PRETOK_CHUNK_SIZE  # 近似;最后一批可能更少
                    pbar.update(1)
                    if pbar.n % 20 == 0:
                        pbar.set_postfix(pieces=len(piece_counts), refresh=False)
                pbar.close()

            if verbose:
                print(
                    f"  [pre-tok] done: ~{n_docs} docs -> {len(piece_counts)} unique pieces "
                    f"({time.time() - t0:.1f}s)",
                    flush=True,
                )

            # 每个预分词片段转为「字节经 b2u 后的字符元组」;相同元组合并频次。
            words: dict[tuple, int] = {}
            for piece, count in piece_counts.items():
                if not piece:
                    continue
                tup = tuple(self.b2u[b] for b in piece.encode("utf-8"))
                if tup:
                    words[tup] = words.get(tup, 0) + count
            del piece_counts

            # 初始:所有相邻对的频次表 + 倒排索引(某对出现在哪些「词元组」里)。
            pair_freqs: defaultdict[tuple, int] = defaultdict(int)
            pair_words: defaultdict[tuple, set] = defaultdict(set)
            for word, freq in words.items():
                for i in range(len(word) - 1):
                    p = (word[i], word[i + 1])
                    pair_freqs[p] += freq
                    pair_words[p].add(word)

            merges: list[tuple[str, str]] = []
            num_merges = vocab_size - 256
            if verbose:
                print(
                    f"  [bpe] target merges = {num_merges}  "
                    f"(initial pairs = {len(pair_freqs)}, unique words = {len(words)})",
                    flush=True,
                )

            t1 = time.time()
            merge_bar = tqdm(
                total=num_merges,
                desc="bpe-merge",
                unit="merge",
                disable=not verbose,
                mininterval=0.5,
                dynamic_ncols=True,
            )
            for step in range(num_merges):
                if not pair_freqs:
                    break
                # 确定性:频次最高者优先;平局则按二元组字典序打破。
                best = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
                best_freq = pair_freqs[best]
                if best_freq <= 0:
                    break

                new_token = best[0] + best[1]
                merges.append(best)
                if new_token not in self.encoder:
                    new_id = len(self.encoder)
                    self.encoder[new_token] = new_id
                    self.decoder[new_id] = new_token

                affected = list(pair_words[best])
                del pair_freqs[best]
                del pair_words[best]

                if pool is not None and len(affected) >= APPLY_PARALLEL_THRESHOLD:
                    # ---- 并行 apply 路径 ----
                    # 按 stride 切块(决定论:不按 hash 不按内容,可重现)。
                    # workers 返回 delta dict; 主进程按 chunk_id 升序合并入全局。
                    N = num_workers
                    chunks = [affected[i::N] for i in range(N)]
                    chunks = [c for c in chunks if c]  # 跳过空 chunk
                    args_list = [
                        (best, new_token, c, {w: words[w] for w in c})
                        for c in chunks
                    ]
                    results = pool.map(_apply_merge_chunk, args_list)

                    for freq_deltas, add_pw, rem_pw, new_words_dict, removed in results:
                        for w in removed:
                            words.pop(w, None)
                        for p, d in freq_deltas.items():
                            pair_freqs[p] += d
                            if pair_freqs[p] <= 0:
                                pair_freqs.pop(p, None)
                        for p, ws in rem_pw.items():
                            s = pair_words.get(p)
                            if s is not None:
                                s.difference_update(ws)
                                if not s:
                                    pair_words.pop(p, None)
                        for w, f in new_words_dict.items():
                            words[w] = words.get(w, 0) + f
                        for p, ws in add_pw.items():
                            pair_words[p].update(ws)
                else:
                    # ---- 单进程 apply 路径(原代码,不动)----
                    for old_word in affected:
                        if old_word not in words:
                            continue
                        freq = words.pop(old_word)
                        # 从所有「非本次合并对」的统计中扣除该旧词元组的贡献。
                        for i in range(len(old_word) - 1):
                            p = (old_word[i], old_word[i + 1])
                            if p == best:
                                continue
                            pair_freqs[p] -= freq
                            if pair_freqs[p] <= 0:
                                pair_freqs.pop(p, None)
                            s = pair_words.get(p)
                            if s is not None:
                                s.discard(old_word)
                                if not s:
                                    pair_words.pop(p, None)
                        # 在原词元组上就地应用合并:连续 best 合并为 new_token。
                        new_word_list = []
                        i = 0
                        last = len(old_word) - 1
                        while i < len(old_word):
                            if i < last and old_word[i] == best[0] and old_word[i + 1] == best[1]:
                                new_word_list.append(new_token)
                                i += 2
                            else:
                                new_word_list.append(old_word[i])
                                i += 1
                        new_word = tuple(new_word_list)
                        words[new_word] = words.get(new_word, 0) + freq
                        # 新词元组产生的相邻对,把频次加回统计与倒排索引。
                        for i in range(len(new_word) - 1):
                            p = (new_word[i], new_word[i + 1])
                            pair_freqs[p] += freq
                            pair_words[p].add(new_word)

                merge_bar.update(1)
                if (step + 1) % log_every == 0:
                    preview = new_token if len(new_token) <= 12 else new_token[:12] + "..."
                    # ascii() 强制 \uXXXX 转义,保证进度条后缀在任意控制台都安全。
                    merge_bar.set_postfix(
                        freq=best_freq,
                        vocab=len(self.encoder),
                        pairs=len(pair_freqs),
                        last=ascii(preview),
                        refresh=False,
                    )
            merge_bar.close()

            self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
            if verbose:
                print(
                    f"  [bpe] done. final vocab_size={len(self.encoder)}  "
                    f"merges={len(merges)}  total {time.time() - t0:.1f}s",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    # ----------------------------------------------------------------- encode
    def _bpe(self, piece_unicode: str) -> list[str]:
        """对单个预分词片段（已是 b2u 后的字符串）按 ``bpe_ranks`` 反复合并，返回子词列表。"""
        if piece_unicode in self._cache:
            return self._cache[piece_unicode]
        word = list(piece_unicode)
        while len(word) >= 2:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            # 选「合并优先级最高」的一对：rank 越小越先合并；未出现的对视为无穷大。
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            new_word = []
            i = 0
            last = len(word) - 1
            while i < len(word):
                if i < last and word[i] == best[0] and word[i + 1] == best[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        self._cache[piece_unicode] = word
        return word

    def encode(self, text: str) -> list[int]:
        """整段文本 -> id 序列：正则切片段，每片段 UTF-8 字节经 b2u 后再 BPE，查表得 id。"""
        ids: list[int] = []
        for piece in self.pat.findall(text):
            if not piece:
                continue
            piece_unicode = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(piece_unicode):
                ids.append(self.encoder[tok])
        return ids

    def decode(self, ids: Iterable[int], errors: str = "replace") -> str:
        """id 序列 -> 文本：子词拼接后经 u2b 还原字节，再 UTF-8 解码。"""
        text_unicode = "".join(self.decoder[i] for i in ids)
        byts = bytes(self.u2b[c] for c in text_unicode)
        return byts.decode("utf-8", errors=errors)

    # --------------------------------------------------------------- save/load
    def save(self, dir_path) -> None:
        """写出 ``vocab.json``（词表）与 ``merges.txt``（按合并顺序的「a b」行）。"""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        with (dir_path / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(self.encoder, f, ensure_ascii=False)
        with (dir_path / "merges.txt").open("w", encoding="utf-8") as f:
            f.write("#version: 0.1\n")
            ranked = sorted(self.bpe_ranks.items(), key=lambda kv: kv[1])
            for (a, b), _ in ranked:
                f.write(f"{a} {b}\n")

    @classmethod
    def load(cls, dir_path) -> "BBPETokenizer":
        """从目录加载 ``vocab.json`` 与 ``merges.txt``，重建 ``encoder`` / ``decoder`` / ``bpe_ranks``。"""
        dir_path = Path(dir_path)
        tok = cls()
        with (dir_path / "vocab.json").open("r", encoding="utf-8") as f:
            enc = json.load(f)
        tok.encoder = {k: int(v) for k, v in enc.items()}
        tok.decoder = {v: k for k, v in tok.encoder.items()}
        merges = []
        with (dir_path / "merges.txt").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                a, b = line.split(" ", 1)
                merges.append((a, b))
        tok.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        return tok


# ---------------------------------------------------------------------------
# 入口：无参数跑内存烟测；``--train`` 在 jsonl 上训练并保存。
# ---------------------------------------------------------------------------
def _smoke() -> None:
    """在内存中训练极小 BBPE，并对多语言/全角/emoji 文本做编解码往返断言。"""
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! Greetings to all programmers writing PyTorch.",
        "你好，世界！这是一个测试。",
        "中文和英文混合 mixed bilingual content。",
        "全角符号：（），。！？；：「」『』",
        "Numbers: 1234567890, decimals 3.14159, units like 42kg.",
        "Emoji and punctuation 👩‍💻 🚀 — should round-trip cleanly.",
    ] * 60

    tok = BBPETokenizer()
    tok.train(iter(corpus), vocab_size=400, verbose=True, log_every=50)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tok.save(tmp)
        tok2 = BBPETokenizer.load(tmp)

    # 断言：中英混排、全角标点、emoji 必须严格可逆。
    test = (
        "你好，World! 中英文混合：PyTorch 训练。"
        "全角符号（），。！？；：「」 emoji 👩‍💻🚀 完毕。"
    )

    for label, t in (("trained", tok), ("loaded", tok2)):
        ids = t.encode(test)
        # 严格解码：往返必须与原文完全一致。
        decoded = t.decode(ids, errors="strict")
        assert decoded == test, (
            f"[{label}] round-trip MISMATCH\n"
            f"  original: {test!r}\n"
            f"  decoded : {decoded!r}"
        )

    ids = tok.encode(test)
    print(f"\n[smoke OK] vocab={tok.vocab_size}, test text -> {len(ids)} tokens")
    print(f"  text: {test}")
    print(f"  head ids: {ids[:25]} ...")


def _iter_jsonl(path: Path, max_samples: int):
    """逐行读取 JSONL，取每条记录的 ``text`` 字段，最多 ``max_samples`` 条非空文本。"""
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if n >= max_samples:
                break
            rec = json.loads(line)
            text = rec.get("text", "")
            if text:
                yield text
                n += 1


def main() -> None:
    """命令行：默认烟测；``--train`` 在指定 jsonl 上训练并保存词表，再做保存/加载 sanity check。"""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train",
        action="store_true",
        help="Train BBPE on data/processed/mix_1to1.jsonl and save to --out-dir. "
             "Heavy: typically run by the user.",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "mix_1to1_3B.jsonl",
    )
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-samples", type=int, default=50_000)
    ap.add_argument(
        "--tokenizer-size",
        default="8k",
        help="Tokenizer variant directory under tokenizer/. e.g. 8k, 32k",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. If omitted, uses tokenizer/<--tokenizer-size>.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU worker processes for pre-tok + merge-apply inner loop. "
             "0 (default, recommended on Windows) = sequential, byte-identical to "
             "historical behavior. >=2 enables multiprocessing but is typically "
             "SLOWER on Windows due to spawn-mode pickle overhead; useful mainly "
             "on Linux fork() platforms. Output is byte-identical across N.",
    )
    args = ap.parse_args()
    out_dir = args.out_dir or (Path(__file__).resolve().parent / args.tokenizer_size)

    if args.train:
        if not args.data.exists():
            print(f"[fatal] data file not found: {args.data}", file=sys.stderr)
            sys.exit(1)
        print(
            f"[train] vocab_size={args.vocab_size}  max_samples={args.max_samples}  "
            f"data={args.data}  num_workers={args.num_workers}",
            flush=True,
        )
        tok = BBPETokenizer()
        tok.train(
            _iter_jsonl(args.data, args.max_samples),
            vocab_size=args.vocab_size,
            num_workers=args.num_workers,
        )
        tok.save(out_dir)
        print(f"[train] saved vocab.json + merges.txt -> {out_dir}")
        # 对落盘文件再加载一次，做短句编解码自检。
        tok2 = BBPETokenizer.load(out_dir)
        sample = "你好 World 👩‍💻"
        ids = tok2.encode(sample)
        assert tok2.decode(ids, errors="strict") == sample
        print(f"[train] save/load round-trip OK ({sample!r} -> {len(ids)} tokens)")
    else:
        _smoke()


if __name__ == "__main__":
    main()
