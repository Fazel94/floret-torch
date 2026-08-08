"""Parity of our floret hashing/word-vector math against CPU floret.

Known upstream quirk (pinned by `test_index0_word_quirk`): floret's
`getWordVector` goes through `Dictionary::getSubwords(int)`, and `initNgrams`
erases the word-table row from the bag only for `i > 0`. The word at vocab
index 0 therefore keeps an extra word row in its mean. That row lives outside
the hash table and is not written to the `.floret` export, so it cannot be
reproduced by any consumer of that file (spaCy included). We use a compact
layout instead: every word is purely hashed, EOS gets a dedicated row.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

floret = pytest.importorskip("floret")

from floret_torch.export import load_floret, word_vector  # noqa: E402
from floret_torch.vocab import Args, build_vocab, char_ngram_entries, word_bag  # noqa: E402

WORDS = [
    "apple", "banana", "the", "quick", "brown", "fox", "jumps", "over",
    "lazy", "dog", "naïve", "über", "東京", "café", "straße", "reading",
    "reader", "reads", "unreadable", "x",
]
OOV = ["zzzqx", "naïveté", "東京都", "unbananable", "qqq"]
BUCKET, DIM, MINN, MAXN, HASHCOUNT = 1000, 16, 2, 3, 2


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("parity")
    corpus = tmp / "tiny.txt"
    rng = random.Random(0)
    corpus.write_text(
        "\n".join(" ".join(rng.choice(WORDS) for _ in range(30))
                  for _ in range(200)) + "\n",
        encoding="utf-8",
    )
    model = floret.train_unsupervised(
        str(corpus), model="cbow", mode="floret", hashCount=HASHCOUNT,
        bucket=BUCKET, minn=MINN, maxn=MAXN, dim=DIM, epoch=1, thread=1,
        minCount=1,
    )
    floret_path = tmp / "vectors.floret"
    model.save_floret_vectors(str(floret_path))
    table, args = load_floret(floret_path)
    return model, table, args, list(model.get_words()), corpus


def test_floret_header_roundtrip(trained):
    _, table, args, _, _ = trained
    assert (args.bucket, args.dim) == (BUCKET, DIM)
    assert (args.minn, args.maxn, args.hashCount) == (MINN, MAXN, HASHCOUNT)
    assert table.shape == (BUCKET, DIM)


@pytest.mark.parametrize("word", WORDS + OOV)
def test_ngram_entries_match_floret(trained, word):
    """Our bag entry strings must equal floret's subword substrings."""
    model, _, args, _, _ = trained
    substrings, _ids = model.get_subwords(word)
    assert list(substrings) == char_ngram_entries(word, args.minn, args.maxn)


@pytest.mark.parametrize("word", WORDS + OOV)
def test_bag_row_ids_match_floret(trained, word):
    """Our hashed row ids must equal floret's, modulo its `nwords` offset.

    This pins the MurmurHash3 seed, the 128->4x uint32 key split, hashCount
    truncation and the bucket modulo all at once.
    """
    model, _, args, words, _ = trained
    _substrings, ids = model.get_subwords(word)
    expected = [int(i) - len(words) for i in ids]
    assert word_bag(word, args) == expected


@pytest.mark.parametrize("word", WORDS + OOV)
def test_word_vector_matches_floret(trained, word):
    """Reconstruction from the `.floret` table reproduces get_word_vector."""
    model, table, args, words, _ = trained
    if word == words[0]:
        pytest.skip("vocab index 0 hits the upstream extra-word-row quirk")
    ours = word_vector(word, table, args)
    ref = model.get_word_vector(word)
    assert np.allclose(ours, ref, atol=1e-4), float(np.abs(ours - ref).max())


def test_index0_word_quirk(trained):
    """Pin the one documented divergence so a future change is deliberate."""
    model, table, args, words, _ = trained
    w0 = words[0]
    ours = word_vector(w0, table, args)
    ref = model.get_word_vector(w0)
    assert not np.allclose(ours, ref, atol=1e-4)
    ok = sum(
        np.allclose(word_vector(w, table, args), model.get_word_vector(w),
                    atol=1e-4)
        for w in words[1:] if w != "</s>"
    )
    assert ok == len([w for w in words[1:] if w != "</s>"])


def test_build_vocab_matches_floret(tmp_path):
    """Vocab membership, counts and frequency order agree with floret."""
    rng = random.Random(1)
    corpus = tmp_path / "c.txt"
    corpus.write_text(
        "\n".join(" ".join(rng.choice(WORDS) for _ in range(20))
                  for _ in range(100)) + "\n",
        encoding="utf-8",
    )
    model = floret.train_unsupervised(
        str(corpus), model="cbow", mode="floret", hashCount=1, bucket=100,
        dim=8, epoch=1, thread=1, minCount=3,
    )
    words, counts, ntokens = build_vocab(corpus, min_count=3)
    ref_words = list(model.get_words())
    ref_counts = [int(c) for c in model.get_words(include_freq=True)[1]]

    assert set(words) == set(ref_words)
    ours = dict(zip(words, counts.tolist()))
    assert [ours[w] for w in ref_words] == ref_counts
    # ntokens counts every token read, including sub-minCount ones and EOS.
    assert ntokens >= int(counts.sum())
    assert ntokens == 100 * 20 + 100  # 20 words/line + one EOS per line


def test_eos_gets_dedicated_row():
    """EOS has no subwords and must map to the single reserved row."""
    args = Args(dim=DIM, minn=MINN, maxn=MAXN, bucket=BUCKET,
                hashCount=HASHCOUNT)
    assert word_bag("</s>", args) == [BUCKET]


def test_hashcount_truncates_prefix():
    """hashCount=k must yield exactly the first k of the 4 derived keys."""
    a4 = Args(bucket=BUCKET, hashCount=4, minn=MINN, maxn=MAXN)
    a2 = Args(bucket=BUCKET, hashCount=2, minn=MINN, maxn=MAXN)
    entries = char_ngram_entries("apple", MINN, MAXN)
    full = word_bag("apple", a4)
    assert len(full) == 4 * len(entries)
    assert word_bag("apple", a2) == [
        h for i, h in enumerate(full) if i % 4 < 2
    ]
