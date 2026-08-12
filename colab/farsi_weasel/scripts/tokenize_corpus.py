"""Tokenize a local WikiExtractor JSONL (or a HuggingFace text dataset) ->
one tokenized sentence per line.

Adapted from explosion/projects `floret_wiki_oscar_vectors`
(scripts/tokenize_resource.py) with the changes needed to run on Colab today:

1. `use_auth_token=` was REMOVED from `datasets.load_dataset`; modern versions
   use `token=`. The upstream script raises TypeError on datasets>=3.
2. Upstream forks a multiprocessing Pool sized for a 16-core box. Colab has
   ~2 vCPUs, and forking a spaCy pipeline per worker there costs more than it
   saves, so tokenization uses `nlp.pipe` in-process.
3. Adds `--max-sents`, so you can cap corpus size to fit a Colab session
   instead of consuming an entire Wikipedia.
4. `--max-texts` caps every input source (JSONL, plain text, HF dataset)
   uniformly, not just the HF streaming path -- the default project.yml
   pipeline now reads a local WikiExtractor JSONL, not the HF dataset.

Output format is what floret/fastText expects: whitespace-separated tokens,
one sentence per line.
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

WS = re.compile(r"\s+")


def iter_texts(args) -> Iterator[str]:
    if args.input_jsonl:
        import srsly
        rows: Iterable = srsly.read_jsonl(args.input_jsonl)
        rows = rows if args.max_texts <= 0 else islice(rows, args.max_texts)
        for row in rows:
            yield row["text"]
        return
    if args.input_text:
        with open(args.input_text, encoding="utf-8") as f:
            lines: Iterable = f
            lines = lines if args.max_texts <= 0 else islice(lines, args.max_texts)
            for line in lines:
                yield line
        return
    from datasets import load_dataset
    ds = load_dataset(args.input_dataset, args.dataset_subset,
                      split=args.dataset_split, streaming=True)
    stream: Iterable = ds if args.max_texts <= 0 else islice(ds,
                                                             args.max_texts)
    for row in stream:
        yield row[args.text_field]


def build_nlp(lang: str):
    import spacy
    try:
        nlp = spacy.blank(lang)
    except ImportError:
        print(f"[warn] {lang} unsupported by spaCy; falling back to xx",
              file=sys.stderr)
        nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 10 ** 8
    return nlp


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("lang")
    p.add_argument("output_file")
    p.add_argument("--input-dataset")
    p.add_argument("--dataset-subset")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--input-jsonl")
    p.add_argument("--input-text")
    p.add_argument("--max-texts", type=int, default=-1)
    p.add_argument("--max-sents", type=int, default=-1,
                   help="stop after this many output sentences (-1 = all)")
    p.add_argument("--batch-size", type=int, default=200)
    args = p.parse_args()

    nlp = build_nlp(args.lang)
    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    texts = (WS.sub(" ", t.strip()) for t in iter_texts(args) if t and t.strip())
    n_sents = n_tokens = 0
    with open(out, "w", encoding="utf-8") as fh:
        for doc in nlp.pipe(texts, batch_size=args.batch_size):
            for sent in doc.sents:
                toks = [t.text for t in sent if not t.is_space]
                if len(toks) < 2:
                    continue           # single-token lines teach nothing
                fh.write(" ".join(toks) + "\n")
                n_sents += 1
                n_tokens += len(toks)
                if args.max_sents > 0 and n_sents >= args.max_sents:
                    print(f"wrote {n_sents} sents / {n_tokens} tokens -> {out}")
                    return 0
            if n_sents and n_sents % 100000 < 2:
                print(f"  {n_sents} sents / {n_tokens} tokens...",
                      file=sys.stderr, flush=True)
    print(f"wrote {n_sents} sents / {n_tokens} tokens -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
