"""CLI: train floret-mode vectors on GPU and export .vec / .floret."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import binarize, open_corpus
from .export import save_floret, save_vec
from .model import FloretModel
from .trainer import Trainer
from .vocab import EOS, Args, build_bags, discard_probs, load_vocab


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="floret_torch.train",
        description="GPU floret vector training (PyTorch).")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True,
                   help="output prefix; writes <prefix>.vec and <prefix>.floret")
    p.add_argument("--model", default="cbow", choices=["cbow", "skipgram"])
    p.add_argument("--mode", default="floret", choices=["floret", "fasttext"])
    p.add_argument("--dim", type=int, default=300)
    p.add_argument("--minn", type=int, default=4)
    p.add_argument("--maxn", type=int, default=5)
    p.add_argument("--hashCount", type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument("--bucket", type=int, default=50000)
    p.add_argument("--epoch", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--ws", type=int, default=5)
    p.add_argument("--neg", type=int, default=5)
    p.add_argument("--minCount", type=int, default=5)
    p.add_argument("--t", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--input-grad", default="fasttext",
                   choices=["fasttext", "normalized"],
                   help="fasttext: un-normalized input-row update (matches "
                        "Model::update; needs a modest batch). normalized: "
                        "1/L-scaled, stable at huge batch but learns slowly")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="row-norm clip on the accumulated input gradient; "
                        "0 disables. Guards against divergence when a word "
                        "recurs many times in one batch")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"],
                   help="storage precision of the vector tables; arithmetic "
                        "stays fp32 either way")
    p.add_argument("--slab", type=int, default=2_000_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workdir", default=None,
                   help="cache dir for corpus.bin/vocab.json "
                        "(default: <output>.work)")
    p.add_argument("--reuse-cache", action="store_true",
                   help="skip binarization if the cache already exists")
    p.add_argument("--seed", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    if ns.mode != "floret":
        raise NotImplementedError(
            "Only --mode floret is implemented; use CPU `floret`/`fasttext` "
            "for the two-table fasttext mode.")
    if ns.seed is not None:
        torch.manual_seed(ns.seed)
        np.random.seed(ns.seed)

    args = Args(dim=ns.dim, minn=ns.minn, maxn=ns.maxn, bucket=ns.bucket,
                hashCount=ns.hashCount, minCount=ns.minCount, t=ns.t,
                ws=ns.ws, neg=ns.neg, epoch=ns.epoch, lr=ns.lr,
                model=ns.model, mode=ns.mode,
                input_grad=ns.input_grad, dtype=ns.dtype,
                max_grad_norm=ns.max_grad_norm)
    out_prefix = Path(ns.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(ns.workdir) if ns.workdir else Path(str(out_prefix) + ".work")

    prep_t = 0.0
    if ns.reuse_cache and (workdir / "vocab.json").exists():
        print(f"reusing cache in {workdir}", file=sys.stderr)
    else:
        t0 = time.time()
        info = binarize(ns.input, workdir, args)
        prep_t = time.time() - t0
        print(f"binarized: {info['nwords']} words, {info['ntokens']} tokens, "
              f"{info['n_kept']} kept, {prep_t:.1f}s", file=sys.stderr)

    words, counts, ntokens, cached = load_vocab(workdir / "vocab.json")
    # Bags depend on minn/maxn/bucket/hashCount, which the cache may predate.
    for k in ("minn", "maxn", "bucket", "hashCount"):
        if getattr(cached, k) != getattr(args, k):
            raise SystemExit(
                f"cache in {workdir} was built with {k}={getattr(cached, k)}, "
                f"but --{k}={getattr(args, k)}; drop --reuse-cache")

    device = torch.device(ns.device)
    t0 = time.time()
    bag_flat, bag_off = build_bags(words, args)
    pdiscard = discard_probs(counts, ntokens, args.t)
    eos_id = words.index(EOS) if EOS in words else None
    bags_t = time.time() - t0
    print(f"bags: {len(bag_flat)} rows refs ({bags_t:.1f}s), "
          f"nwords={len(words)}, eos_id={eos_id}", file=sys.stderr)

    model = FloretModel(args, len(words), counts, device)
    trainer = Trainer(model, args, bag_flat, bag_off, pdiscard, eos_id,
                      device, batch=ns.batch, slab=ns.slab)
    corpus = open_corpus(workdir)

    t0 = time.time()
    state = trainer.train(corpus, ntokens)
    train_t = time.time() - t0
    print(f"trained {state.updates} pairs in {train_t:.1f}s "
          f"(loss {state.loss:.4f})", file=sys.stderr)

    table = model.input_table()
    t0 = time.time()
    save_vec(str(out_prefix) + ".vec", words, table, args)
    save_floret(str(out_prefix) + ".floret", table, args)
    print(f"exported {out_prefix}.vec / {out_prefix}.floret "
          f"({time.time() - t0:.1f}s)", file=sys.stderr)
    print(f"TIMING preprocess={prep_t:.1f}s bags={bags_t:.1f}s "
          f"train={train_t:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
