"""Hand-written SGNS forward/backward over a Bloom subword table.

Autograd was measured at 139ms/batch for the output side plus a 33ms dense
optimizer sweep, because it allocates and zeroes dense (rows, dim) gradient
buffers every step and then updates every row whether or not it was touched.
The SGNS gradient is analytic and touches few rows, so we compute it directly
and scatter with `index_add_`. This also reproduces fastText's exact update
order (`Model::update` / `BinaryLogisticLoss::binaryLogistic`):

    alpha  = lr * (label - sigmoid(w_o . h))
    grad  += alpha * w_o        <- uses the PRE-update w_o
    w_o   += alpha * h
    W_r   += grad               for every row r of the input bag

Note the input rows get `grad` un-normalized by bag size: fastText only
divides when `normalizeGradient_` is set, which happens for supervised
training, not for sg/cbow.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .vocab import Args

NEGATIVE_TABLE_SIZE = 10_000_000


def build_negative_table(counts: np.ndarray, device: torch.device) -> Tensor:
    """fastText `NegativeSamplingLoss`: row i repeated ~ count_i**0.5.

    The exponent is 0.5, not word2vec's 0.75 (floret src/loss.cc:148-161).
    """
    c = np.power(counts.astype(np.float64), 0.5)
    reps = np.floor(c * NEGATIVE_TABLE_SIZE / c.sum()).astype(np.int64)
    table = np.repeat(np.arange(len(counts), dtype=np.int32), reps)
    return torch.from_numpy(table).to(device)


class FloretModel:
    """Input table = hashed buckets (+1 EOS row); output table = per word."""

    def __init__(self, args: Args, nwords: int, counts: np.ndarray,
                 device: torch.device):
        self.args = args
        self.nwords = nwords
        self.device = device
        # fp16 halves the traffic of the gathers that dominate this workload.
        # Only STORAGE is reduced: every logit and gradient below is computed
        # in fp32 on the small gathered slices, because an SGNS update is
        # ~1e-5 against weights ~3e-3 and fp16's ulp there is ~4e-6.
        self.dtype = torch.float16 if args.dtype == "fp16" else torch.float32
        # +1 row: EOS has no subwords. Rows 0..bucket-1 are what gets exported.
        self.wi = torch.empty(args.bucket + 1, args.dim, device=device,
                              dtype=self.dtype)
        self.wi.uniform_(-1.0 / args.dim, 1.0 / args.dim)   # fastText init
        self.wo = torch.zeros(nwords, args.dim, device=device,
                              dtype=self.dtype)
        self.neg_table = build_negative_table(counts, device)

    def sample_negatives(self, n: int) -> Tensor:
        idx = torch.randint(0, self.neg_table.numel(), (n, self.args.neg),
                            device=self.device)
        return self.neg_table[idx].long()

    def word_sums(self, ids: Tensor, offsets: Tensor) -> Tensor:
        """Summed subword vector per bag (callers divide by the bag length)."""
        return F.embedding_bag(ids, self.wi, offsets, mode="sum")

    def sgns_update(self, hidden: Tensor, targets: Tensor,
                    lr: float) -> Tuple[Tensor, float]:
        """Apply the output-side update in place; return (grad wrt hidden, loss).

        `hidden` is (B, dim). One positive plus `neg` sampled negatives per
        row, handled as a single (B, 1+neg) block.
        """
        b = targets.numel()
        negatives = self.sample_negatives(b)
        out_ids = torch.cat([targets.unsqueeze(1), negatives], dim=1)
        w_out = self.wo[out_ids].float()                        # (B, 1+neg, d)

        logits = torch.einsum("bd,bkd->bk", hidden, w_out)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        # A negative colliding with its target is dropped; fastText resamples
        # until distinct, collisions are ~1e-4 either way.
        mask = (out_ids != targets.unsqueeze(1))
        mask[:, 0] = True

        alpha = lr * (labels - torch.sigmoid(logits)) * mask   # (B, 1+neg)
        grad_hidden = torch.einsum("bk,bkd->bd", alpha, w_out)
        d_out = alpha.unsqueeze(-1) * hidden.unsqueeze(1)      # (B, 1+neg, d)
        self.wo.index_add_(0, out_ids.reshape(-1),
                           d_out.reshape(-1, self.args.dim).to(self.dtype))

        # -log sigmoid(±logit), summed like fastText's per-example loss.
        loss = (F.softplus(-(2.0 * labels - 1.0) * logits) * mask).sum()
        return grad_hidden, float(loss)

    def scatter_input(self, ids: Tensor, row_grad: Tensor) -> None:
        """Add `row_grad` to every input row named by `ids`."""
        self.wi.index_add_(0, ids, row_grad.to(self.dtype))

    def input_table(self) -> np.ndarray:
        """The full input table: bucket hash rows plus the EOS row at `bucket`.

        `save_floret` slices off the EOS row, which the format cannot carry;
        `save_vec` needs it, because `</s>` is a normal vocab entry whenever
        the corpus has line breaks.
        """
        return self.wi.detach().float().cpu().numpy()
