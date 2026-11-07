"""Exact search over the whole corpus: BM25, dense inner product, compressed embeddings and rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BM25:
    """bm25s with English stopwords and Snowball stemming."""

    def __init__(self, texts: list[str]):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.retriever = bm25s.BM25()
        self.retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=self.stemmer, show_progress=False), show_progress=False)
        self.n_docs = len(texts)

    def search(self, queries: list[str], k: int) -> np.ndarray:
        tokens = self._bm25s.tokenize(queries, stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)
        # A query whose every term is a stopword would crash bm25s; give it one unknown token instead.
        tokens = [t if t else ["<empty>"] for t in tokens]
        results, _ = self.retriever.retrieve(tokens, k=min(k, self.n_docs), show_progress=False)
        return np.asarray(results)


def normalise(x: np.ndarray) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def encode(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return normalise(model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False))


def topk(queries: np.ndarray, docs: np.ndarray, k: int, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product, best first."""
    k = min(k, docs.shape[0])
    idx = np.empty((len(queries), k), dtype=np.int64)
    val = np.empty((len(queries), k), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        scores = queries[start : start + chunk] @ docs.T
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        idx[start : start + chunk] = np.take_along_axis(part, order, axis=1)
        val[start : start + chunk] = np.take_along_axis(part_scores, order, axis=1)
    return idx, val


def truncate(x: np.ndarray, dim: int) -> np.ndarray:
    """Keep the first ``dim`` coordinates and renormalise, as a Matryoshka-trained model is meant to be used."""
    return normalise(x[:, :dim])


@dataclass
class Int8Codec:
    """Per-dimension scalar quantisation to int8, with ranges taken from a calibration sample of documents."""

    low: np.ndarray
    high: np.ndarray

    @classmethod
    def fit(cls, sample: np.ndarray) -> Int8Codec:
        return cls(sample.min(axis=0), sample.max(axis=0))

    def encode(self, x: np.ndarray) -> np.ndarray:
        scaled = (np.clip(x, self.low, self.high) - self.low) / np.maximum(self.high - self.low, 1e-12)
        return np.round(scaled * 255 - 128).astype(np.int8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return (self.low + (codes.astype(np.float32) + 128) / 255 * (self.high - self.low)).astype(np.float32)

    def search(self, queries: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
        """Asymmetric: the float query against dequantised documents. Quantising the query too, and taking the dot
        product of raw codes, adds an offset term that favours some documents regardless of the query."""
        return topk(queries, self.decode(codes), k)[0]


def binary_encode(x: np.ndarray) -> np.ndarray:
    """One bit per dimension (the sign), packed 8 to a byte."""
    return np.packbits(x > 0, axis=1)


def binary_search(queries: np.ndarray, codes: np.ndarray, k: int, rescore: int = 4, chunk: int = 16) -> np.ndarray:
    """Hamming search over packed bits, then rescore the best ``rescore * k`` with the float query against the doc's signs."""
    q_bits = binary_encode(queries)
    candidates_k = min(rescore * k, codes.shape[0])
    out = np.empty((len(queries), min(k, codes.shape[0])), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        block = q_bits[start : start + chunk]
        distance = np.bitwise_count(block[:, None, :] ^ codes[None, :, :]).sum(axis=2, dtype=np.int32)
        cand = np.argpartition(distance, candidates_k - 1, axis=1)[:, :candidates_k]
        signs = np.unpackbits(codes[cand], axis=2, count=queries.shape[1]).astype(np.float32) * 2 - 1
        scores = np.einsum("qd,qcd->qc", queries[start : start + chunk], signs)
        order = np.argsort(-scores, axis=1, kind="stable")[:, : out.shape[1]]
        out[start : start + chunk] = np.take_along_axis(cand, order, axis=1)
    return out


def rrf(rankings: list[np.ndarray], k: int, constant: int = 60) -> np.ndarray:
    """Reciprocal rank fusion (Cormack et al., 2009): score = sum of 1 / (constant + rank)."""
    out = np.empty((rankings[0].shape[0], k), dtype=np.int64)
    for row in range(rankings[0].shape[0]):
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking[row]):
                scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (constant + rank + 1)
        best = sorted(scores, key=lambda d: (-scores[d], d))[:k]
        out[row, : len(best)] = best
        out[row, len(best) :] = -1
    return out
