"""Corpus binarization: text -> uint32 token-id stream + vocab.json."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

from .vocab import Args, build_vocab, save_vocab, tokenize


def binarize(input_path: str | Path, out_dir: str | Path, args: Args) -> dict:
    """Build the vocab and write `corpus.bin` (uint32 ids) + `vocab.json`.

    Out-of-vocab tokens are dropped, matching `Dictionary::getLine`, which
    skips them (they still counted toward the raw token total, which is
    recorded as `ntokens` in vocab.json for the lr schedule).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    words, counts, ntokens = build_vocab(input_path, args.minCount)
    word2id = {w: i for i, w in enumerate(words)}
    corpus_path = out_dir / "corpus.bin"
    n_written = 0
    buf: List[int] = []
    with open(input_path, "r", encoding="utf-8", errors="replace") as fp, \
            open(corpus_path, "wb") as out:
        for tok in tokenize(fp):
            wid = word2id.get(tok)
            if wid is None:
                continue
            buf.append(wid)
            if len(buf) >= (1 << 20):
                np.asarray(buf, dtype=np.uint32).tofile(out)
                n_written += len(buf)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=np.uint32).tofile(out)
            n_written += len(buf)
    save_vocab(out_dir / "vocab.json", words, counts, ntokens, args)
    return {
        "nwords": len(words),
        "ntokens": ntokens,
        "n_kept": n_written,
        "seconds": time.time() - t0,
        "corpus": str(corpus_path),
    }


def open_corpus(out_dir: str | Path) -> np.memmap:
    path = Path(out_dir) / "corpus.bin"
    return np.memmap(path, dtype=np.uint32, mode="r")
