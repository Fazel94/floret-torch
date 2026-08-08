"""Sanity-check Farsi vectors: nearest neighbours + OOV behaviour.

floret has no fixed vocabulary, so neighbours are ranked over the words that
actually occur in the training corpus. We recover that list from the packaged
pipeline's vocab; any word, including unseen ones, still gets a vector.
"""

from __future__ import annotations

import sys

import numpy as np
import spacy

# book / city / woman / water / good / football / university / wrote
PROBES = ["کتاب", "تهران", "زن", "آب", "خوب", "فوتبال", "دانشگاه", "نوشت"]
# Deliberately unseen/inflected forms: floret should still place these well.
OOV_PROBES = ["کتاب‌هایشان", "تهرانی‌ها", "دانشگاهیان"]


def main(path: str) -> int:
    nlp = spacy.load(path)
    vocab_words = [w for w in nlp.vocab.strings if w.strip()]
    if not vocab_words:
        print("empty vocab; is this a floret pipeline?")
        return 1

    mat = np.vstack([nlp.vocab.get_vector(w) for w in vocab_words])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.maximum(norms, 1e-9)

    def neighbours(word: str, k: int = 8):
        v = nlp.vocab.get_vector(word)
        n = np.linalg.norm(v)
        if n < 1e-9:
            return []
        sims = unit @ (v / n)
        top = np.argpartition(-sims, k + 1)[:k + 1]
        top = top[np.argsort(-sims[top])]
        return [(vocab_words[i], float(sims[i]))
                for i in top if vocab_words[i] != word][:k]

    print(f"vectors: {mat.shape[0]} words x {mat.shape[1]} dims\n")
    for tag, probes in (("in-corpus", PROBES), ("unseen", OOV_PROBES)):
        print(f"--- {tag} probes ---")
        for w in probes:
            nb = neighbours(w)
            joined = ", ".join(f"{x}({s:.2f})" for x, s in nb) or "<no vector>"
            print(f"  {w}: {joined}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "/content/vectors/fa_vectors"))
