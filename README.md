# floret-torch

[![tests](https://github.com/Fazel94/floret-torch/actions/workflows/tests.yml/badge.svg)](https://github.com/Fazel94/floret-torch/actions/workflows/tests.yml)

Train [floret](https://github.com/explosion/floret) word vectors on a GPU, ~20× faster than
CPU floret at the same quality, and load them straight into spaCy. floret is fastText with
a compact subword table: each character n-gram is hashed to `hashCount` rows of a fixed
table of `bucket` rows (Bloom-filter style), so any word, even an unseen one, gets a vector
from its n-grams. floret-torch reimplements training in PyTorch and writes the same
`.floret` and `.vec` files.

Use it when CPU floret training is the slow step: a large corpus, or a Colab session with a
GPU but only two CPU cores.

## Results

text8 (17M tokens of English Wikipedia), cbow,
`dim=300 bucket=50000 hashCount=2 minn=4 maxn=5 neg=5`.
Quality is Spearman ρ on WordSim-353 (WS353), a word-similarity benchmark.

| epochs | run | hardware | train time | WS353 |
|---:|---|---|---:|---:|
| 1 | CPU floret | i7-7500U, 4 threads | 188.1s | 0.2875 |
| 1 | floret-torch | GeForce 940MX (sm_50) | 178.5s | 0.3582 |
| 3 | CPU floret | i7-7500U, 4 threads | 593.1s | 0.4738 |
| 3 | floret-torch | Colab T4 | ~30s | 0.4740 |

At three epochs the vectors match CPU floret (0.4740 vs 0.4738). Training is ~20× faster
than the 4-core laptop, about 1M training pairs/sec on a T4.

The 940MX result is the odd one. It wins with half the memory bandwidth of the CPU it beats
(16 GB/s vs ~34 GB/s dual-channel DDR4).

Hashing is byte-exact against CPU floret, so a word maps to the same table rows in both.
`tests/test_parity.py` checks this against a live `floret` model: the MurmurHash3 seed,
the split of each 128-bit hash into four 32-bit keys, `hashCount` truncation, n-gram
extraction and the bucket modulo.

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

Add the `weasel` extra (`'.[bench,weasel]'`) for the Farsi pipeline below.

## Usage

Get text8, then train:

```bash
mkdir -p data
curl -sL -o data/text8.zip http://mattmahoney.net/dc/text8.zip
unzip -o -q -d data data/text8.zip

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

`--dtype fp16` stores both tables in fp16 but computes every logit and gradient in fp32.
That split is deliberate: a single update is ~1e-5 against weights ~3e-3, and fp16's
smallest step at that size (its ulp) is ~4e-6. Storing in fp16 is safe; accumulating in
fp16 stops learning once `lr` decays.

Expect ~10–15%, no more; this workload is bound by table lookups, not arithmetic. Measured
on a T4: matmul 5.02→0.55ms (9×) but `embedding_bag` fwd+bwd only 28.27→24.29ms (1.16×),
and `embedding_bag` is the hot path. On sm_50 GPUs fp16 is *slower* than fp32 (no tensor
cores, and no native paired-fp16 math below sm_53).

## How it gets the speed

No custom CUDA kernel: Triton and `torch.compile` need a GPU of compute capability 7.0
(sm_70) or newer, the dev box's 940MX is 5.0 (sm_50), and it has no `nvcc`. Two changes do
the work.

**Per-batch word deduplication.** A word's input vector is the sum of its subword rows (its
"bag"), and every occurrence of the word in a batch has the same bag. So each bag is
gathered once and indexed. This is exact, not an approximation: the table is constant within
a forward pass. On text8, 65536 context tokens collapse to ~3900 unique words.

**Hand-written gradients.** Both cbow and skip-gram train with negative sampling (SGNS in
the code): score the true output word against a few random ones. With autograd this cost
139ms/batch on the output side plus a 33ms dense optimizer sweep, because autograd
allocates and zeroes dense
`(rows, dim)` gradient buffers each step and then updates all 121k rows. The SGNS gradient
has a closed form and touches few rows, so it is computed directly and scattered with
`index_add_`, in the same order as fastText's `Model::update`. For hidden vector `h` (the
averaged input bag), output-word vector `w_o` and label 1 (true word) or 0 (negative):

```
alpha  = lr * (label - sigmoid(w_o . h))
grad  += alpha * w_o        # uses the PRE-update w_o
w_o   += alpha * h
W_r   += grad               # every row r of the input bag
```

The real code is in [`floret_torch/model.py`](floret_torch/model.py). Measured effect on
full text8: 235.7s → 178.5s.

## Two correctness traps

**The input-row gradient must NOT be divided by bag size.** fastText divides only in
supervised mode. Skipping it looks like a bug and it is load-bearing: a cbow window averages
~240 subword rows, so dividing by the bag size `L` makes the gradient ~240× too small to get
learning started while `w_o` starts at zero. With `1/L` the model silently produces
plausible-looking vectors that score near zero (WS353 −0.12, every pairwise cosine 1.00).
`--input-grad normalized` selects that variant for experiments; don't train with it. The
default `fasttext` is correct.

**Batch size is a correctness knob.** The undivided update adds one full gradient per
occurrence, so a word appearing *k* times in a batch moves *k*× as far, and oversized
batches diverge outright. Keep `--batch` in 4096–16384 (default 8192). `--max-grad-norm`
(default 1.0) bounds per-batch row movement and turns divergence into slow learning;
training aborts with an actionable message if the loss goes non-finite anyway.

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

[Weasel](https://github.com/explosion/weasel) is spaCy's workflow runner: a `project.yml`
lists named steps and `weasel run` executes them. `colab/farsi_weasel/` adapts explosion's
[`floret_wiki_oscar_vectors`](https://github.com/explosion/projects/tree/v3/pipelines/floret_wiki_oscar_vectors)
project, which downloads Wikipedia plus OSCAR (a large web-crawl corpus), tokenizes them,
trains floret on CPU and packages a spaCy model. To run on Colab it changes four things:

- Vectors train on the GPU with floret-torch (`train-vectors-gpu`). CPU floret stays
  available as `train-vectors-cpu` for comparison.
- OSCAR is dropped: `unshuffled_deduplicated_fa` is gated and fails unauthenticated. The
  corpus is Persian Wikipedia alone, the `fawiki` dump run through WikiExtractor
  `--no-templates`, with `cleanup-corpus` deleting the dump and extracted JSONL once
  tokenizing is done.
- Tokenizing runs in-process with `nlp.pipe`, not a multiprocessing pool sized for a 16-core
  box.
- Paths point at `/content`, not `/scratch`, and process counts suit ~2 vCPUs.

```bash
weasel run all colab/farsi_weasel   # Wikipedia → GPU vectors → spaCy package
weasel run cpu colab/farsi_weasel   # CPU floret instead; no packaging
```

## Credits

Algorithm and file formats follow [explosion/floret](https://github.com/explosion/floret)
and fastText. The Weasel project derives from
[explosion/projects](https://github.com/explosion/projects) `pipelines/floret_wiki_oscar_vectors`.

## Citation

If you use floret-torch in your work, please cite:

```bibtex
@software{fazeli2026florettorch,
  author  = {Fazeli, Mohammad},
  title   = {floret-torch: GPU training for floret vectors in PyTorch},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Fazel94/floret-torch},
  note    = {Contact: kiyarash@nlogn.ir}
}
```

## License

MIT, see [LICENSE](LICENSE). floret, fastText and explosion/projects are MIT too.
