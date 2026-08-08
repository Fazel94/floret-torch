"""CPU floret baseline with parameters matched to the GPU trainer."""

import argparse
import time

import floret

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--model", default="cbow")
p.add_argument("--dim", type=int, default=300)
p.add_argument("--epoch", type=int, default=1)
p.add_argument("--thread", type=int, default=4)
p.add_argument("--lr", type=float, default=0.05)
ns = p.parse_args()

t0 = time.time()
m = floret.train_unsupervised(
    ns.input, model=ns.model, mode="floret", hashCount=2, bucket=50000,
    minn=4, maxn=5, dim=ns.dim, epoch=ns.epoch, thread=ns.thread, lr=ns.lr,
    ws=5, neg=5, minCount=5, t=1e-4,
)
elapsed = time.time() - t0
m.save_vectors(ns.output + ".vec")
m.save_floret_vectors(ns.output + ".floret")
print(f"CPU_TRAIN_SECONDS {elapsed:.1f} model={ns.model} dim={ns.dim} "
      f"epoch={ns.epoch} thread={ns.thread}")
