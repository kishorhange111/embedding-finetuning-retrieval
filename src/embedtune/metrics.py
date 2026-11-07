"""Per-query retrieval metrics over the full corpus, with bootstrap intervals over queries.

nDCG uses linear gain on the judged relevance, as pytrec_eval and the BEIR and MTEB leaderboards do.
"""

from __future__ import annotations

import numpy as np


def per_query(ranked: np.ndarray, positives: list[set[int]], gains: list[dict[int, int]] | None = None) -> dict[str, np.ndarray]:
    """nDCG@10, MRR@10 and Recall@100 for each query. ``ranked`` holds document indices, best first."""
    discounts = 1.0 / np.log2(np.arange(2, 12))
    n = len(positives)
    ndcg, mrr, recall100 = np.zeros(n), np.zeros(n), np.zeros(n)
    for row, relevant in enumerate(positives):
        gain = gains[row] if gains else dict.fromkeys(relevant, 1)
        top10 = ranked[row, :10]
        dcg = sum(gain.get(int(d), 0) * discounts[i] for i, d in enumerate(top10))
        ideal = sorted(gain.values(), reverse=True)[:10]
        ndcg[row] = dcg / sum(g * discounts[i] for i, g in enumerate(ideal))
        hits = [i for i, d in enumerate(top10) if int(d) in relevant]
        mrr[row] = 1.0 / (hits[0] + 1) if hits else 0.0
        recall100[row] = len(relevant & set(ranked[row, :100].tolist())) / len(relevant)
    return {"ndcg@10": ndcg, "mrr@10": mrr, "recall@100": recall100}


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of ``a - b`` over the same queries, with a 95 % interval. An interval that excludes 0 is a real difference."""
    low, high = bootstrap_ci(a - b, n_boot, seed)
    return {"diff": round(float((a - b).mean()), 4), "low": round(low, 4), "high": round(high, 4)}


def summarise(values: dict[str, np.ndarray]) -> dict:
    out: dict = {"queries": int(len(next(iter(values.values()))))}
    for name, v in values.items():
        low, high = bootstrap_ci(v)
        out[name] = {"mean": round(float(v.mean()), 4), "low": round(low, 4), "high": round(high, 4)}
    return out
