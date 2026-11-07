"""Hard-negative mining with the base model.

The closest non-positive documents make the strongest negatives, but on a corpus with incomplete judgements many of them are
unlabelled positives. The positive-aware filter (Moreira et al., 2024, "NV-Retriever") drops any candidate scoring above a
fraction of the positive's own score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrieval import topk


@dataclass(frozen=True)
class MiningConfig:
    first_rank: int = 0  # 0-based, among non-positive results
    last_rank: int = 30
    max_ratio: float | None = None  # keep a candidate only if its score < max_ratio * positive score
    seed: int = 0


@dataclass
class Triplets:
    query: list[int]
    positive: list[int]
    negative: list[int]
    stats: dict


def pick_positives(positives: list[set[int]], seed: int = 0) -> list[int]:
    """One judged positive per query, chosen at random, so every variant trains on the same pairs."""
    rng = np.random.default_rng(seed)
    return [int(rng.choice(sorted(p))) for p in positives]


def mine(query_emb: np.ndarray, doc_emb: np.ndarray, positives: list[set[int]], cfg: MiningConfig, chosen: list[int] | None = None) -> Triplets:
    """A (query, positive, negative) triplet per judged pair, or only for the ``chosen`` positive of each query.

    Candidates exclude every judged positive of the query, not just the one being paired.
    """
    rng = np.random.default_rng(cfg.seed)
    depth = cfg.last_rank + max(len(p) for p in positives)
    idx, scores = topk(query_emb, doc_emb, depth)
    out = Triplets([], [], [], {"pairs": 0, "filtered_candidates": 0, "candidates": 0, "random_fallback": 0})
    for qi, relevant in enumerate(positives):
        ranked = [(int(d), float(s)) for d, s in zip(idx[qi], scores[qi]) if int(d) not in relevant]
        window = ranked[cfg.first_rank : cfg.last_rank]
        for p in [chosen[qi]] if chosen is not None else sorted(relevant):
            out.stats["pairs"] += 1
            limit = cfg.max_ratio * float(query_emb[qi] @ doc_emb[p]) if cfg.max_ratio is not None else np.inf
            choices = [d for d, s in window if s < limit]
            out.stats["candidates"] += len(window)
            out.stats["filtered_candidates"] += len(window) - len(choices)
            if choices:
                negative = int(rng.choice(choices))
            else:
                out.stats["random_fallback"] += 1
                negative = int(rng.integers(doc_emb.shape[0]))
                while negative in relevant:
                    negative = int(rng.integers(doc_emb.shape[0]))
            out.query.append(qi)
            out.positive.append(p)
            out.negative.append(negative)
    return out
