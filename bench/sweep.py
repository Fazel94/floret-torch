"""Batch-size / lr sweep: measures the speed-vs-quality tradeoff.

Batch size is the dominant quality knob here. The loss is summed, so batch
size does not change the effective step magnitude -- it changes how many
optimization steps happen and how stale each one is. fastText performs one
update per example; a huge batch performs a handful of updates per epoch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.eval import load_ws353, normalize, ws353_spearman  # noqa: E402
from floret_torch.data import binarize, open_corpus  # noqa: E402
from floret_torch.export import word_vector  # noqa: E402
from floret_torch.model import FloretModel  # noqa: E402
from floret_torch.trainer import Trainer  # noqa: E402
from floret_torch.vocab import (EOS, Args, build_bags,  # noqa: E402
                                discard_probs, load_vocab)


def evaluate(words, table, args, ws_rows) -> tuple:
    mat = np.vstack([word_vector(w, table, args) for w in words])
    unit = normalize(mat)
    idx = {w: i for i, w in enumerate(words)}
    rho, used, _ = ws353_spearman(ws_rows, idx, unit)
    mean_cos = float((unit[:1000] @ unit[:1000].T).mean())
    return rho, used, mean_cos


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--workdir", default="out/sweep.work")
    p.add_argument("--batches", default="1024,4096,16384,65536")
    p.add_argument("--lrs", default="0.05")
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--model", default="cbow")
    p.add_argument("--ws353", default="data/wordsim353/raw.txt")
    p.add_argument("--dtypes", default="fp32")
    ns = p.parse_args()

    base = Args(dim=ns.dim, model=ns.model, epoch=ns.epoch)
    if not (Path(ns.workdir) / "vocab.json").exists():
        binarize(ns.input, ns.workdir, base)
    words, counts, ntokens, _ = load_vocab(Path(ns.workdir) / "vocab.json")
    corpus = open_corpus(ns.workdir)
    ws_rows = load_ws353(ns.ws353)
    dev = torch.device("cuda")
    eos_id = words.index(EOS) if EOS in words else None

    print(f"corpus={ns.input} nwords={len(words)} ntokens={ntokens} "
          f"model={ns.model} epoch={ns.epoch}")
    print(f"{'dtype':>5} {'batch':>8} {'lr':>6} {'steps':>8} {'sec':>7} "
          f"{'loss':>8} {'WS353':>8} {'meancos':>8}")
    for dt in ns.dtypes.split(","):
        for lr in (float(x) for x in ns.lrs.split(",")):
          for batch in (int(x) for x in ns.batches.split(",")):
            args = Args(dim=ns.dim, model=ns.model, epoch=ns.epoch, lr=lr,
                        dtype=dt)
            bag_flat, bag_off = build_bags(words, args)
            pdisc = discard_probs(counts, ntokens, args.t)
            torch.manual_seed(0)
            model = FloretModel(args, len(words), counts, dev)
            tr = Trainer(model, args, bag_flat, bag_off, pdisc, eos_id, dev,
                         batch=batch)
            t0 = time.time()
            st = tr.train(corpus, ntokens, verbose=False)
            el = time.time() - t0
            rho, used, mc = evaluate(words, model.input_table(), args, ws_rows)
            print(f"{dt:>5} {batch:>8} {lr:>6} {st.loss_n:>8} {el:>7.1f} "
                  f"{st.loss:>8.4f} {rho:>8.4f} {mc:>8.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
