"""GPU training loop: subsample -> segment -> pair-gen -> batched SGNS.

Two ideas carry the performance:

1. Per-batch word deduplication. Every occurrence of a word in a batch has
   the same input vector, so its subword bag is gathered once and indexed.
   This is exact (the table is constant within a forward pass) and removes
   the dominant cost -- measured 65536 context tokens collapsing to ~3900
   unique words on text8.
2. Hand-written gradients scattered with `index_add_` (see model.py), so no
   dense gradient buffer is allocated and no optimizer sweeps untouched rows.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .model import FloretModel
from .vocab import MAX_LINE_SIZE, Args


def gather_bags(words: Tensor, bag_flat: Tensor,
                bag_off: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Concatenate the CSR bags of `words`, in order.

    Returns (flat row ids, per-word bag length, owner index per row). The
    ragged gather is vectorized: each output slot maps back to
    `bag_off[w] + (slot - start_of_that_word)`.
    """
    starts = bag_off[words]
    lens = bag_off[words + 1] - starts
    out_off = torch.cumsum(lens, 0) - lens          # exclusive prefix sum
    total = int(lens.sum())
    slot = torch.arange(total, device=words.device)
    owner = torch.repeat_interleave(
        torch.arange(words.numel(), device=words.device), lens)
    idx = starts[owner] + (slot - out_off[owner])
    return bag_flat[idx], lens, owner


def segment_keys(kept: Tensor, eos_id: Optional[int]) -> Tensor:
    """Segment id per kept token: breaks after EOS and every MAX_LINE_SIZE.

    Context windows must never cross a segment boundary (fastText `getLine`
    returns a single line, capped at 1024 tokens).
    """
    m = kept.numel()
    device = kept.device
    prev_eos = torch.zeros(m, dtype=torch.long, device=device)
    if eos_id is not None:
        prev_eos[1:] = (kept[:-1] == eos_id).long()
    seg = torch.cumsum(prev_eos, 0)
    nseg = int(seg[-1]) + 1
    first = torch.searchsorted(seg, torch.arange(nseg, device=device))
    pos_in_seg = torch.arange(m, device=device) - first[seg]
    return seg * (m // MAX_LINE_SIZE + 2) + pos_in_seg // MAX_LINE_SIZE


def generate_pairs(kept: Tensor, segkey: Tensor,
                   ws: int) -> Tuple[Tensor, Tensor]:
    """(center_position, context_word) for one slab, grouped by center.

    Per fastText, each center draws its own boundary b ~ U{1..ws} and keeps
    offsets d with 0 < |d| <= b that stay inside the same segment.
    """
    m = kept.numel()
    device = kept.device
    if m < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    b = torch.randint(1, ws + 1, (m,), device=device)
    offsets = torch.cat([torch.arange(-ws, 0, device=device),
                         torch.arange(1, ws + 1, device=device)])
    pos = torch.arange(m, device=device).unsqueeze(1)     # (m, 1)
    nb = pos + offsets                                    # (m, 2*ws)
    valid = offsets.abs().unsqueeze(0) <= b.unsqueeze(1)
    valid &= (nb >= 0) & (nb < m)
    nb = nb.clamp_(0, m - 1)
    valid &= segkey[nb] == segkey.unsqueeze(1)
    center_pos = pos.expand(m, offsets.numel())[valid]
    context = kept[nb[valid]]
    return center_pos.contiguous(), context.contiguous()


@dataclass
class TrainState:
    tokens_seen: int = 0
    updates: int = 0
    loss_sum: float = 0.0
    loss_n: int = 0

    @property
    def loss(self) -> float:
        return self.loss_sum / max(self.loss_n, 1)


class Trainer:
    def __init__(self, model: FloretModel, args: Args, bag_flat: np.ndarray,
                 bag_off: np.ndarray, pdiscard: np.ndarray,
                 eos_id: Optional[int], device: torch.device,
                 batch: int = 65536, slab: int = 4_000_000):
        self.model = model
        self.args = args
        self.device = device
        self.batch = batch
        self.slab = slab
        self.eos_id = eos_id
        self.bag_flat = torch.from_numpy(bag_flat).to(device)
        self.bag_off = torch.from_numpy(bag_off).to(device)
        self.pdiscard = torch.from_numpy(pdiscard).to(device)

    def _lr(self, progress: float) -> float:
        return self.args.lr * max(0.0, 1.0 - progress)

    def _clip(self, grad: Tensor) -> Tensor:
        """Bound each row's accumulated update to `max_grad_norm`."""
        limit = self.args.max_grad_norm
        if limit and limit > 0:
            norms = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad.mul_((limit / norms).clamp_(max=1.0))
        return grad

    def _run_skipgram(self, kept: Tensor, center_pos: Tensor,
                      context: Tensor, state: TrainState,
                      sched: Callable[[float], float]) -> None:
        """Input = center word's bag (mean), target = context word."""
        n = center_pos.numel()
        for s in range(0, n, self.batch):
            lr = sched(s / n)
            e = min(s + self.batch, n)
            uniq, inv = torch.unique(kept[center_pos[s:e]],
                                     return_inverse=True)
            ids, lens, owner = gather_bags(uniq, self.bag_flat, self.bag_off)
            offsets = torch.cumsum(lens, 0) - lens
            sums = self.model.word_sums(ids, offsets).float()
            inv_len = 1.0 / lens.unsqueeze(1).float()
            hidden = sums * inv_len

            grad_h, loss = self.model.sgns_update(hidden[inv], context[s:e],
                                                  lr)
            grad_u = torch.zeros_like(sums).index_add_(0, inv, grad_h)
            if self.args.input_grad == "normalized":
                grad_u *= inv_len
            self.model.scatter_input(ids, self._clip(grad_u)[owner])

            state.loss_sum += loss / (e - s)
            state.loss_n += 1
            state.updates += e - s

    def _run_cbow(self, kept: Tensor, center_pos: Tensor, context: Tensor,
                  state: TrainState,
                  sched: Callable[[float], float]) -> None:
        """Input = flat n-gram bag of the whole window, target = center word.

        hidden_c = sum_w S_w / sum_w L_w over the window's context words w,
        which equals fastText's mean over the concatenated n-gram bag even
        though different words contribute different numbers of rows.
        """
        n = center_pos.numel()
        s = 0
        while s < n:
            lr = sched(s / n)
            e = min(s + self.batch, n)
            if e < n:
                # Never split one center's window across two batches.
                e = int(torch.searchsorted(center_pos, center_pos[e - 1] + 1))
            cpos, ctx = center_pos[s:e], context[s:e]
            uniq_c, inv_c = torch.unique_consecutive(cpos, return_inverse=True)
            uniq_w, inv_w = torch.unique(ctx, return_inverse=True)
            ids, lens, owner = gather_bags(uniq_w, self.bag_flat, self.bag_off)
            offsets = torch.cumsum(lens, 0) - lens
            sums = self.model.word_sums(ids, offsets).float()

            c = uniq_c.numel()
            num = torch.zeros(c, self.args.dim, device=self.device)
            num.index_add_(0, inv_c, sums[inv_w])
            den = torch.zeros(c, device=self.device)
            den.index_add_(0, inv_c, lens[inv_w].float())
            den = den.clamp_min(1.0)
            hidden = num / den.unsqueeze(1)

            grad_h, loss = self.model.sgns_update(hidden, kept[uniq_c], lr)
            if self.args.input_grad == "normalized":
                # d(hidden_c)/d(row) = 1/D_c for every row in the window.
                grad_h /= den.unsqueeze(1)
            grad_u = torch.zeros_like(sums).index_add_(0, inv_w,
                                                       grad_h[inv_c])
            self.model.scatter_input(ids, self._clip(grad_u)[owner])

            state.loss_sum += loss / c
            state.loss_n += 1
            state.updates += c
            s = e

    def train(self, corpus: np.memmap, ntokens_raw: int,
              verbose: bool = True) -> TrainState:
        state = TrainState()
        total = max(self.args.epoch * ntokens_raw, 1)
        t0 = time.time()
        next_log = 0.0
        run = (self._run_skipgram if self.args.model == "skipgram"
               else self._run_cbow)
        for _ in range(self.args.epoch):
            for start in range(0, corpus.shape[0], self.slab):
                chunk = np.asarray(corpus[start:start + self.slab])
                tokens = torch.from_numpy(chunk.astype(np.int64)).to(
                    self.device)
                keep = (torch.rand(tokens.numel(), device=self.device)
                        < self.pdiscard[tokens])
                kept = tokens[keep]
                base, n_slab = state.tokens_seen, int(tokens.numel())
                if kept.numel() >= 2:
                    segkey = segment_keys(kept, self.eos_id)
                    center_pos, context = generate_pairs(kept, segkey,
                                                         self.args.ws)
                    if center_pos.numel():
                        run(kept, center_pos, context, state,
                            lambda f: self._lr((base + f * n_slab) / total))
                state.tokens_seen += n_slab
                if not np.isfinite(state.loss):
                    raise RuntimeError(
                        f"loss diverged to {state.loss}: the batch is too "
                        f"large for this vocabulary ({self.batch} pairs). "
                        f"Halve --batch, or lower --max-grad-norm.")

                progress = state.tokens_seen / total
                if verbose and progress >= next_log:
                    next_log = progress + 0.01
                    el = time.time() - t0
                    print(f"\rprogress {progress * 100:5.1f}% | "
                          f"words/sec {state.tokens_seen / max(el, 1e-9):9.0f} "
                          f"| loss {state.loss:.4f} | {el:6.1f}s",
                          end="", file=sys.stderr, flush=True)
        if verbose:
            print(file=sys.stderr)
        return state
