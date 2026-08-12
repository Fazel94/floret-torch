# floret-torch

[![tests](https://github.com/Fazel94/floret-torch/actions/workflows/tests.yml/badge.svg)](https://github.com/Fazel94/floret-torch/actions/workflows/tests.yml)

GPU training for [floret](https://github.com/explosion/floret) vectors (fastText with
subword n-grams hashed into a compact Bloom table), implemented in PyTorch, exporting
spaCy-compatible `.floret` and `.vec` tables.

Hashing is byte-exact against CPU floret: the MurmurHash3 seed, the 128-bit to 4×uint32
key split, `hashCount` truncation, n-gram extraction and the bucket modulo are all pinned
by `tests/test_parity.py` against a live `floret` model.

## Results

text8 (17M tokens), cbow, `dim=300 bucket=50000 hashCount=2 minn=4 maxn=5 neg=5`.
Quality is Spearman ρ on WordSim-353.

| epochs | run | hardware | train time | WS353 |
|---:|---|---|---:|---:|
| 1 | CPU floret | i7-7500U, 4 threads | 188.1s | 0.2875 |
| 1 | floret-torch | GeForce 940MX (sm_50) | 178.5s | 0.3582 |
| 3 | CPU floret | i7-7500U, 4 threads | 593.1s | 0.4738 |
| 3 | floret-torch | Colab T4 | ~30s | 0.4740 |

At three epochs the vectors match CPU floret (0.4740 vs 0.4738) and arrive ~20× faster than
the 4-core box, ~60× faster than Colab's 2 vCPUs: ~1M training pairs/sec on a T4 against
~20k on the CPU. The 940MX result is the odd one: it wins with half the memory bandwidth of
the CPU it beats (16 GB/s vs ~34 GB/s dual-channel DDR4).

## How it gets the speed

No custom CUDA kernel; none is possible on sm_50 (Triton and `torch.compile` need sm_70+,
and there is no `nvcc` on the dev box). Two changes do the work.

**Per-batch word deduplication.** Every occurrence of a word in a batch has the same input
vector, so its subword bag is gathered once and indexed. This is exact, not an
approximation: the table is constant within a forward pass. On text8, 65536 context tokens
collapse to ~3900 unique words.

**Hand-written gradients.** Autograd cost 139ms/batch on the output side plus a 33ms dense
optimizer sweep, because it allocates and zeroes dense `(rows, dim)` gradient buffers each
step and then updates all 121k rows. The SGNS gradient is analytic and touches few rows, so
it is computed directly and scattered with `index_add_`, reproducing fastText's update
order from `Model::update`:

```
alpha  = lr * (label - sigmoid(w_o . h))
grad  += alpha * w_o        # uses the PRE-update w_o
w_o   += alpha * h
W_r   += grad               # every row r of the input bag
```

Measured effect on full text8: 235.7s → 178.5s.

## Two correctness traps

**The input-row gradient must NOT be divided by bag size.** fastText normalizes only in
supervised mode. It looks like a bug and it is load-bearing: a cbow window averages ~240
subword rows, so the `1/L` gradient is ~240× too small to bootstrap while `wo` starts at
zero. With `1/L` the model silently produces plausible-looking vectors that score near zero
(WS353 −0.12, every pairwise cosine 1.00). `--input-grad normalized` selects the
consistent-but-useless variant; the default `fasttext` is correct.

**Batch size is a correctness knob.** The un-normalized update sums one full gradient per
occurrence, so a word appearing *k* times in a batch moves *k*× as far, and oversized
batches diverge outright. Keep `--batch` in 4096–16384 (default 8192). `--max-grad-norm`
(default 1.0) bounds per-batch row movement and turns divergence into slow learning;
training aborts with an actionable message if the loss goes non-finite anyway.

## Install

Needs Python >=3.9 and PyTorch (any build; CUDA only for the GPU path). torch is
deliberately not a declared dependency, so pip never pulls a second multi-GB copy.

```bash
pip install git+https://github.com/Fazel94/floret-torch
```

From a clone, reusing an already-installed torch:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[bench]'
.venv/bin/python -m pytest tests/test_parity.py -q      # hashing parity gate
```

## Usage

```bash
python -m floret_torch.train \
  --input data/text8 --output out/vectors \
  --model cbow --dim 300 --minn 4 --maxn 5 --hashCount 2 --bucket 50000 \
  --epoch 3 --lr 0.05 --batch 8192 --dtype fp32 --device cuda
```

Writes `out/vectors.floret` and `out/vectors.vec`. Load into spaCy with:

```bash
python -m spacy init vectors en out/vectors.floret out/pipeline --mode floret
```

`--mode fasttext` (the two-table layout) raises `NotImplementedError`; use CPU floret for
that layout.

### fp16

`--dtype fp16` stores both tables in fp16 while keeping every logit and gradient in fp32 on
the gathered slices. That split is deliberate: an SGNS update is ~1e-5 against weights
~3e-3, where fp16's ulp is ~4e-6, so storing fp16 is safe but accumulating in fp16 stops
learning once `lr` decays.

Expect ~10–15%, no more; this workload is gather-bound. Measured on a T4: matmul
5.02→0.55ms (9×) but `embedding_bag` fwd+bwd only 28.27→24.29ms (1.16×), and
`embedding_bag` is the hot path. On sm_50 fp16 is *slower* than fp32 (no tensor cores, no
native `half2` below sm_53).

## Layout

```
floret_torch/     vocab (hashing) · data (binarize) · model (SGNS) ·
                  trainer (batching) · export (.vec/.floret) · train (CLI)
tests/            hashing + word-vector parity against CPU floret
bench/            cpu_baseline · eval (WS353 + neighbours) · sweep
colab/            T4 notebook, plain-command version, bundle script
colab/farsi_weasel/   Weasel project: Persian Wikipedia → floret → spaCy
```

## Farsi / Weasel pipeline

`colab/farsi_weasel/` adapts explosion's `floret_wiki_oscar_vectors` project to run on
Colab. Three changes were required:

- OSCAR is gated now (`unshuffled_deduplicated_fa` fails unauthenticated), so the corpus is
  Persian Wikipedia alone: the `fawiki` dump run through WikiExtractor `--no-templates`,
  with `cleanup-corpus` deleting the dump and extracted JSONL once tokenizing is done.
- `use_auth_token=` was removed from `datasets.load_dataset`; upstream's
  `tokenize_resource.py` raises `TypeError` on `datasets>=3`.
- Paths point at `/content` and process counts suit ~2 vCPUs, not a 16-core box with
  `/scratch`.

```bash
weasel run all colab/farsi_weasel       # tokenize → GPU vectors → spaCy
weasel run cpu colab/farsi_weasel       # same, CPU floret for comparison
```

## Credits

Algorithm and file formats follow [explosion/floret](https://github.com/explosion/floret)
and fastText. The Weasel project derives from
[explosion/projects](https://github.com/explosion/projects) `pipelines/floret_wiki_oscar_vectors`.

## License

MIT, see [LICENSE](LICENSE). floret, fastText and explosion/projects are MIT too.
