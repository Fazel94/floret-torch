"""Quality gate: nearest neighbours, CPU/GPU agreement, WS353 Spearman."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floret_torch.export import load_floret, load_vec, word_vector  # noqa: E402
from floret_torch.vocab import load_vocab  # noqa: E402

PROBES = ["king", "paris", "music", "three", "cat"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with average ranks for ties (no scipy dependency)."""
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ranks within tie groups
        sx = x[order]
        i = 0
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


def neighbours(word: str, words: List[str], index: Dict[str, int],
               unit: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    i = index.get(word)
    if i is None:
        return []
    sims = unit @ unit[i]
    sims[i] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(words[j], float(sims[j])) for j in top]


def load_ws353(path: Optional[str]) -> List[Tuple[str, str, float]]:
    """Parse a wordsim353 `combined.csv`/tab file: word1, word2, score."""
    if not path or not Path(path).exists():
        return []
    rows: List[Tuple[str, str, float]] = []
    for line in Path(path).read_text(encoding="utf-8",
                                     errors="replace").splitlines():
        parts = line.replace(",", "\t").split("\t")
        if len(parts) < 3:
            continue
        try:
            rows.append((parts[0].strip().lower(), parts[1].strip().lower(),
                         float(parts[2])))
        except ValueError:
            continue          # header line
    return rows


def ws353_spearman(rows, index, unit) -> Tuple[float, int, int]:
    gold, pred = [], []
    for w1, w2, score in rows:
        i, j = index.get(w1), index.get(w2)
        if i is None or j is None:
            continue
        gold.append(score)
        pred.append(float(unit[i] @ unit[j]))
    if len(gold) < 10:
        return float("nan"), len(gold), len(rows)
    return spearman(np.array(pred), np.array(gold)), len(gold), len(rows)


def load_any(path: str, vocab: Optional[str]) -> Tuple[List[str], np.ndarray]:
    """Load a `.vec`, or reconstruct word vectors from a `.floret` table.

    A `.floret` file holds only the hash table, so it needs a vocab to know
    which words to materialize; any `.vec` or a vocab.json supplies that.
    """
    if not path.endswith(".floret"):
        return load_vec(path)
    if not vocab:
        raise SystemExit(f"{path} is a hash table; pass --vocab "
                         "<vocab.json|reference .vec> to name the words")
    table, args = load_floret(path)
    if vocab.endswith(".json"):
        words = load_vocab(vocab)[0]
    else:
        words = load_vec(vocab)[0]
    return words, np.vstack([word_vector(w, table, args) for w in words])


def report(tag: str, vec_path: str, ws_rows, topn: int, vocab=None):
    words, mat = load_any(vec_path, vocab)
    index = {w: i for i, w in enumerate(words)}
    unit = normalize(mat)
    print(f"\n=== {tag}: {vec_path} ({len(words)} words, dim {mat.shape[1]}) ===")
    for p in PROBES:
        nb = neighbours(p, words, index, unit, topn)
        if nb:
            print(f"  {p:>8}: " + ", ".join(f"{w}({s:.2f})" for w, s in nb))
        else:
            print(f"  {p:>8}: <not in vocab>")
    if ws_rows:
        rho, used, total = ws353_spearman(ws_rows, index, unit)
        print(f"  WS353 spearman: {rho:.4f}  ({used}/{total} pairs covered)")
    return words, index, unit


def overlap(a, b, topn: int, n_words: int) -> float:
    """Mean top-n neighbour overlap over the n_words most frequent words."""
    words_a, index_a, unit_a = a
    words_b, index_b, unit_b = b
    shared = [w for w in words_a[:n_words] if w in index_b]
    if not shared:
        return float("nan")
    total = 0.0
    for w in shared:
        na = {x for x, _ in neighbours(w, words_a, index_a, unit_a, topn)}
        nb = {x for x, _ in neighbours(w, words_b, index_b, unit_b, topn)}
        total += len(na & nb) / topn
    return total / len(shared)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vec", required=True, help="vectors under test (.vec)")
    p.add_argument("--compare", help="reference .vec (e.g. the CPU baseline)")
    p.add_argument("--ws353", default="data/wordsim353/raw.txt")
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--overlap-words", type=int, default=50)
    p.add_argument("--vocab", help="vocab.json or .vec naming the words when "
                                   "--vec/--compare is a .floret table")
    ns = p.parse_args()

    ws_rows = load_ws353(ns.ws353)
    if not ws_rows:
        print(f"(no WS353 file at {ns.ws353}; skipping that metric)")

    a = report("under test", ns.vec, ws_rows, ns.topn, ns.vocab)
    if ns.compare:
        b = report("reference", ns.compare, ws_rows, ns.topn, ns.vocab)
        ov = overlap(a, b, ns.topn, ns.overlap_words)
        print(f"\ntop-{ns.topn} neighbour overlap over {ns.overlap_words} "
              f"most frequent words: {ov:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
