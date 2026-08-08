"""Writers/readers for the `.vec` and `.floret` vector tables."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .vocab import MURMURHASH_SEED, Args, word_bag


def word_vector(word: str, table: np.ndarray, args: Args) -> np.ndarray:
    """Mean of the word's bag rows (fasttext.cc `getWordVector`).

    Accepts either the full training table (bucket + EOS row) or a table
    parsed back from a `.floret` file (bucket rows only). Rows past the end
    are dropped, so `</s>` -- whose only row is the EOS row -- yields a zero
    vector when read from a `.floret` file, which cannot represent it.
    """
    rows = [r for r in word_bag(word, args) if r < len(table)]
    if not rows:
        return np.zeros(table.shape[1], dtype=np.float32)
    return table[rows].mean(axis=0)


def _format_rows(table: np.ndarray, labels: Sequence[str]) -> List[str]:
    """`label v0 .. vN` lines.

    One flat `map` over the raveled array beats per-row f-string joins
    (measured 7.2s vs 23.2s for a 50000x300 table).
    """
    fmt = "%.5g".__mod__
    dim = table.shape[1]
    strs = list(map(fmt, np.ascontiguousarray(table, dtype=np.float32).ravel()))
    return [f"{lab} " + " ".join(strs[i * dim:(i + 1) * dim]) + "\n"
            for i, lab in enumerate(labels)]


def save_vec(path: str | Path, words: Sequence[str], table: np.ndarray,
             args: Args) -> None:
    """`.vec`: header `nwords dim`, then `word v0 .. vN` in vocab order."""
    vecs = np.vstack([word_vector(w, table, args) for w in words])
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(words)} {args.dim}\n")
        f.writelines(_format_rows(vecs, words))


def save_floret(path: str | Path, table: np.ndarray, args: Args) -> None:
    """`.floret`: header `bucket dim minn maxn hashCount hashSeed BOW EOW`.

    Only rows 0..bucket-1 are written; the EOS row at index `bucket` is
    internal (fasttext.cc `saveFloretVectors` writes exactly the hash table).
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"{args.bucket} {args.dim} {args.minn} {args.maxn} "
            f"{args.hashCount} {MURMURHASH_SEED} < >\n"
        )
        f.writelines(_format_rows(table[:args.bucket],
                                  [str(i) for i in range(args.bucket)]))


def load_floret(path: str | Path) -> Tuple[np.ndarray, Args]:
    """Parse a `.floret` table (ours or floret's) into (table, args)."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().split()
        bucket, dim, minn, maxn, hash_count = (int(x) for x in header[:5])
        args = Args(dim=dim, minn=minn, maxn=maxn, bucket=bucket,
                    hashCount=hash_count)
        table = np.zeros((bucket, dim), dtype=np.float32)
        for line in f:
            parts = line.split()
            if len(parts) != dim + 1:
                continue
            table[int(parts[0])] = np.asarray(parts[1:], dtype=np.float32)
    return table, args


def load_vec(path: str | Path) -> Tuple[List[str], np.ndarray]:
    """Parse a `.vec` file into (words, matrix).

    Splits on arbitrary whitespace: floret writes a trailing space before
    each newline, so `split(" ")` would yield a spurious empty final field
    and silently drop every row.
    """
    words: List[str] = []
    rows: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        _, dim = (int(x) for x in f.readline().split())
        for line in f:
            parts = line.split()
            if len(parts) != dim + 1:
                continue
            words.append(parts[0])
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    return words, np.vstack(rows) if rows else np.zeros((0, dim), np.float32)
