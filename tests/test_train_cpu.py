"""End-to-end CPU training smoke: CLI -> .vec/.floret with usable vectors."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from floret_torch.export import load_floret, load_vec, word_vector  # noqa: E402
from floret_torch.train import main  # noqa: E402

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]


def test_cpu_training_exports_usable_vectors(tmp_path):
    corpus = tmp_path / "corpus.txt"
    rng = np.random.default_rng(0)
    corpus.write_text(
        "\n".join(" ".join(rng.choice(WORDS, size=20)) for _ in range(400)),
        encoding="utf-8")
    prefix = tmp_path / "vectors"

    rc = main([
        "--input", str(corpus), "--output", str(prefix),
        "--dim", "16", "--bucket", "500", "--minn", "3", "--maxn", "4",
        "--minCount", "1", "--epoch", "2", "--batch", "512", "--neg", "3",
        "--device", "cpu", "--seed", "0",
    ])
    assert rc == 0

    words, mat = load_vec(str(prefix) + ".vec")
    assert set(WORDS) <= set(words)
    assert mat.shape == (len(words), 16)

    table, args = load_floret(str(prefix) + ".floret")
    assert table.shape == (500, 16)
    assert np.isfinite(table).all()
    assert np.abs(table).max() > 0            # training actually moved rows

    v = word_vector("alpha", table, args)     # .floret reconstruction path
    assert np.isfinite(v).all() and np.linalg.norm(v) > 0
