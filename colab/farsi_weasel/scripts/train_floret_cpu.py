"""CPU floret baseline, flag-compatible with the GPU trainer."""

import argparse
import time

import floret

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--model", default="cbow")
p.add_argument("--dim", type=int, default=300)
p.add_argument("--minn", type=int, default=5)
p.add_argument("--maxn", type=int, default=5)
p.add_argument("--neg", type=int, default=10)
p.add_argument("--epoch", type=int, default=5)
p.add_argument("--hashcount", type=int, default=2)
p.add_argument("--bucket", type=int, default=50000)
p.add_argument("--mincount", type=int, default=20)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--thread", type=int, default=2)
ns = p.parse_args()

t0 = time.time()
m = floret.train_unsupervised(
    ns.input, model=ns.model, mode="floret", dim=ns.dim, minn=ns.minn,
    maxn=ns.maxn, neg=ns.neg, epoch=ns.epoch, hashCount=ns.hashcount,
    bucket=ns.bucket, minCount=ns.mincount, lr=ns.lr, thread=ns.thread,
)
elapsed = time.time() - t0
m.save_vectors(ns.output + ".vec")
m.save_floret_vectors(ns.output + ".floret")
print(f"CPU_TRAIN_SECONDS {elapsed:.1f} thread={ns.thread} epoch={ns.epoch}")
