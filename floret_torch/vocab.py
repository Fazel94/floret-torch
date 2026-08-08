"""Tokenization, vocabulary, subsampling and floret subword hashing.

Semantics mirror explosion/floret `src/dictionary.cc` exactly (see plan).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, List, Sequence

import mmh3
import numpy as np

EOS = "</s>"
BOW = "<"
EOW = ">"
MURMURHASH_SEED = 2166136261
MAX_LINE_SIZE = 1024
# fastText separators: ' ' \n \r \t \v \f \0 ; '\n' additionally emits EOS.
_SEPS = frozenset(" \n\r\t\v\f\0")


@dataclass
class Args:
    """Training/vocab settings shared by preprocessing, training and export."""

    dim: int = 300
    minn: int = 4
    maxn: int = 5
    bucket: int = 50000
    hashCount: int = 2
    minCount: int = 5
    t: float = 1e-4
    ws: int = 5
    neg: int = 5
    epoch: int = 1
    lr: float = 0.1
    model: str = "cbow"
    mode: str = "floret"
    # "fasttext": add the gradient to every input row un-normalized, exactly
    # as Model::update does for sg/cbow. This is load-bearing: a cbow window
    # averages ~240 subword rows, so a 1/L-normalized gradient is ~240x too
    # small to bootstrap while wo starts at zero. "normalized" is the
    # consistent d(mean)/d(row) = 1/L variant: stable at huge batch but it
    # barely learns.
    input_grad: str = "fasttext"
    # Storage precision of the vector tables. Arithmetic is always fp32.
    dtype: str = "fp32"
    # Row-norm clip on the accumulated input gradient. The fastText-style
    # un-normalized update sums one full gradient per occurrence, so a word
    # appearing k times in a batch moves k times as far -- on a small vocab
    # (or an oversized batch) that diverges outright. Clipping bounds the
    # per-batch movement of frequent words without touching rare ones.
    # 0 disables.
    max_grad_norm: float = 1.0


def tokenize(fp) -> Iterator[str]:
    """Yield whitespace-separated tokens; each '\\n' yields the EOS token.

    Streams in chunks so a 100MB corpus never materializes as one token list.
    """
    pending: List[str] = []
    while True:
        chunk = fp.read(1 << 20)
        if not chunk:
            break
        for ch in chunk:
            if ch in _SEPS:
                if pending:
                    yield "".join(pending)
                    pending.clear()
                if ch == "\n":
                    yield EOS
            else:
                pending.append(ch)
    if pending:
        yield "".join(pending)


def build_vocab(path: str | Path, min_count: int) -> tuple[List[str], np.ndarray, int]:
    """Return (words sorted by count desc, counts, ntokens-before-threshold)."""
    counter: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as fp:
        for tok in tokenize(fp):
            counter[tok] += 1
    ntokens = sum(counter.values())
    # fastText sorts by count desc; ties broken by insertion order there, by
    # word here (deterministic and irrelevant to training).
    items = [(w, c) for w, c in counter.items() if c >= min_count]
    items.sort(key=lambda wc: (-wc[1], wc[0]))
    if not items:
        raise ValueError("Empty vocabulary. Try a smaller minCount value.")
    words = [w for w, _ in items]
    counts = np.array([c for _, c in items], dtype=np.int64)
    return words, counts, ntokens


def discard_probs(counts: np.ndarray, ntokens: int, t: float) -> np.ndarray:
    """fastText `initTableDiscard`: pdiscard = sqrt(t/f) + t/f, f = count/ntokens.

    A token is kept iff uniform(0,1) < pdiscard (values may exceed 1 -> always kept).
    """
    f = counts.astype(np.float64) / float(ntokens)
    return (np.sqrt(t / f) + t / f).astype(np.float32)


def char_ngram_entries(word: str, minn: int, maxn: int) -> List[str]:
    """floret bag entries for `word`: the padded word plus its char ngrams.

    Mirrors `Dictionary::computeSubwords`: ngram lengths are counted in
    characters over the padded string, and a length-1 ngram is skipped when it
    starts at the first character or ends at the last character.
    """
    s = BOW + word + EOW
    entries = [s]
    n_chars = len(s)
    for i in range(n_chars):
        for n in range(1, maxn + 1):
            j = i + n
            if j > n_chars:
                break
            if n < minn:
                continue
            if n == 1 and (i == 0 or j == n_chars):
                continue
            entries.append(s[i:j])
    return entries


def hash_entry(entry: str, bucket: int, hash_count: int) -> List[int]:
    """MurmurHash3_x64_128 -> 4 uint32 keys -> first `hash_count` bucket rows."""
    h1, h2 = mmh3.hash64(entry.encode("utf-8"), seed=MURMURHASH_SEED, signed=False)
    keys = (h1 & 0xFFFFFFFF, h1 >> 32, h2 & 0xFFFFFFFF, h2 >> 32)
    return [keys[i] % bucket for i in range(hash_count)]


def word_bag(word: str, args: Args) -> List[int]:
    """Input-table row indices whose mean is the embedding of `word`.

    EOS has no subwords; it gets the dedicated row `bucket` (fastText uses its
    word row, which our compact layout replaces).
    """
    if word == EOS:
        return [args.bucket]
    out: List[int] = []
    for entry in char_ngram_entries(word, args.minn, args.maxn):
        out.extend(hash_entry(entry, args.bucket, args.hashCount))
    return out


def build_bags(words: Sequence[str], args: Args) -> tuple[np.ndarray, np.ndarray]:
    """CSR encoding of every vocab word's bag: (flat ids int64, offsets int64)."""
    offsets = np.zeros(len(words) + 1, dtype=np.int64)
    flat: List[int] = []
    for i, w in enumerate(words):
        bag = word_bag(w, args)
        flat.extend(bag)
        offsets[i + 1] = len(flat)
    return np.asarray(flat, dtype=np.int64), offsets


def save_vocab(path: str | Path, words: Sequence[str], counts: np.ndarray,
               ntokens: int, args: Args) -> None:
    payload = {
        "words": list(words),
        "counts": counts.tolist(),
        "ntokens": int(ntokens),
        "args": asdict(args),
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_vocab(path: str | Path) -> tuple[List[str], np.ndarray, int, Args]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        payload["words"],
        np.array(payload["counts"], dtype=np.int64),
        int(payload["ntokens"]),
        Args(**payload["args"]),
    )
