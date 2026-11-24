"""Edge-case, regression, and data-validation tests for embedtune."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from embedtune.config import IndexConfig
from embedtune.data import doc_text, from_frames
from embedtune.metrics import bootstrap_ci, per_query, summarise
from embedtune.mining import MiningConfig, mine
from embedtune.retrieval import BM25, Int8Codec, binary_encode, binary_search, normalise, rrf, topk, truncate
from embedtune.serve import RetrievalIndex


def test_normalise_handles_zero_vector_without_nans():
    x = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    out = normalise(x)
    assert np.isfinite(out).all()
    assert np.allclose(out[0], 0.0)
    assert np.allclose(np.linalg.norm(out[1]), 1.0)


def test_topk_when_k_exceeds_corpus_size():
    docs = normalise(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    q = normalise(np.array([[1.0, 0.2]], dtype=np.float32))
    idx, vals = topk(q, docs, k=10)
    assert idx.shape == (1, 2)
    assert vals.shape == (1, 2)
    assert idx[0, 0] == 0


def test_int8_codec_handles_constant_dimension_and_clipping():
    sample = np.array([[0.5, -1.0], [0.5, 1.0], [0.5, 0.0]], dtype=np.float32)
    codec = Int8Codec.fit(sample)
    extreme = np.array([[5.0, 10.0], [-5.0, -10.0]], dtype=np.float32)
    codes = codec.encode(extreme)
    assert codes.dtype == np.int8
    decoded = codec.decode(codes)
    assert np.isfinite(decoded).all()
    assert np.allclose(decoded[:, 0], 0.5, atol=1e-4)


def test_binary_encode_and_search_non_multiple_of_eight_dims():
    rng = np.random.default_rng(42)
    docs = normalise(rng.normal(size=(40, 13)).astype(np.float32))
    q = docs[:3]
    codes = binary_encode(docs)
    assert codes.shape == (40, 2)
    ranked = binary_search(q, codes, k=5, rescore=2)
    assert ranked.shape == (3, 5)
    for i in range(3):
        assert ranked[i, 0] == i


def test_rrf_pads_with_minus_one_when_union_smaller_than_k():
    a = np.array([[0, 1]])
    b = np.array([[1, 0]])
    fused = rrf([a, b], k=5)
    assert fused.shape == (1, 5)
    assert set(fused[0, :2].tolist()) == {0, 1}
    assert (fused[0, 2:] == -1).all()


def test_mining_falls_back_to_random_when_all_candidates_filtered():
    docs = normalise(np.ones((10, 4), dtype=np.float32))
    q = normalise(np.ones((1, 4), dtype=np.float32))
    triplets = mine(q, docs, [{0}], MiningConfig(first_rank=0, last_rank=5, max_ratio=0.5, seed=0), chosen=[0])
    assert triplets.stats["random_fallback"] == 1
    assert triplets.negative[0] != 0


def test_beir_split_skips_unjudged_and_missing_doc_queries():
    corpus = pd.DataFrame({"_id": ["d1", "d2"], "text": ["alpha", "beta"]})
    queries = pd.DataFrame({"_id": ["q1", "q2", "q3"], "text": ["query 1", "query 2", "query 3"]})
    qrels = {
        "test": pd.DataFrame(
            {
                "query-id": ["q1", "q2", "q_missing", "q3"],
                "corpus-id": ["d1", "d_missing", "d1", "d2"],
                "score": ["2", "1", "1", "0"],
            }
        )
    }
    data = from_frames("edge", corpus, queries, qrels)
    qids, texts, positives = data.split("test")
    assert qids == ["q1"]
    assert texts == ["query 1"]
    assert positives == [{0}]


def test_doc_text_handles_missing_and_whitespace_titles():
    assert doc_text(None, "  body text  ") == "body text"
    assert doc_text("   ", "body") == "body"
    assert doc_text("Title", "Body") == "Title Body"


def test_metrics_summarise_and_bootstrap_determinism():
    ranked = np.array([[0, 1, 2], [2, 1, 0]])
    values = per_query(ranked, [{0}, {0}])
    s1 = summarise(values)
    s2 = summarise(values)
    assert s1 == s2
    assert s1["queries"] == 2
    assert 0.0 <= s1["ndcg@10"]["low"] <= s1["ndcg@10"]["mean"] <= s1["ndcg@10"]["high"] <= 1.0
    assert bootstrap_ci(values["ndcg@10"], n_boot=200, seed=7) == bootstrap_ci(values["ndcg@10"], n_boot=200, seed=7)


def test_bm25_single_doc_and_stopword_query():
    bm25 = BM25(["financial statement dividend yield"])
    ranked = bm25.search(["the and of", "dividend"], k=5)
    assert ranked.shape == (2, 1)
    assert ranked[1, 0] == 0


def test_retrieval_index_matryoshka_truncation_and_custom_encoder():
    def fake_encoder(texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            v = np.zeros(16, dtype=np.float32)
            v[0] = 1.0 if "tax" in t else 0.0
            v[1] = 1.0 if "bread" in t else 0.0
            v[2] = 0.1
            out.append(v)
        return np.asarray(out, dtype=np.float32)

    idx = RetrievalIndex(
        doc_ids=["d1", "d2"],
        doc_texts=["dividend tax rules", "bread recipe"],
        config=IndexConfig(mode="dense", truncate_dim=8),
        encoder=fake_encoder,
    )
    assert idx.dim == 8
    hits = idx.search("tax question", top_k=2)
    assert hits[0].doc_id == "d1"
    with pytest.raises(ValueError):
        truncate(np.ones((2, 4), dtype=np.float32), 4)
        idx.search("tax", mode="dense", query_vector=[1.0, 0.0])
