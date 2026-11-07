import numpy as np
import pandas as pd
import pytest

from embedtune.data import from_frames
from embedtune.metrics import paired_diff, per_query
from embedtune.mining import MiningConfig, mine, pick_positives
from embedtune.retrieval import BM25, Int8Codec, binary_encode, binary_search, normalise, rrf, topk, truncate


def test_ndcg_matches_a_hand_computed_value():
    # Relevant docs 3 and 7; ranked at positions 1 and 3.
    ranked = np.array([[3, 1, 7, 2, 4, 5, 6, 8, 9, 10] + list(range(11, 101))])
    m = per_query(ranked, [{3, 7}])
    dcg = 1 + 1 / np.log2(4)
    ideal = 1 + 1 / np.log2(3)
    assert m["ndcg@10"][0] == pytest.approx(dcg / ideal)
    assert m["mrr@10"][0] == 1.0
    assert m["recall@100"][0] == 1.0


def test_metrics_score_zero_when_nothing_relevant_is_found():
    ranked = np.arange(100)[None, :]
    m = per_query(ranked, [{500}])
    assert m["ndcg@10"][0] == 0 and m["mrr@10"][0] == 0 and m["recall@100"][0] == 0


def test_graded_gain_is_linear():
    ranked = np.array([[1, 2] + list(range(10, 108))])
    m = per_query(ranked, [{1, 2}], gains=[{1: 1, 2: 2}])
    dcg = 1 + 2 / np.log2(3)
    ideal = 2 + 1 / np.log2(3)
    assert m["ndcg@10"][0] == pytest.approx(dcg / ideal)


def test_paired_diff_interval_excludes_zero_only_for_a_real_gap():
    rng = np.random.default_rng(0)
    a = rng.uniform(size=500)
    assert paired_diff(a + 0.1, a)["low"] > 0
    noise = paired_diff(a, a + rng.normal(0, 0.01, 500))
    assert noise["low"] < 0 < noise["high"]


def test_topk_is_exact_and_sorted():
    rng = np.random.default_rng(0)
    docs, q = normalise(rng.normal(size=(1000, 16))), normalise(rng.normal(size=(20, 16)))
    idx, val = topk(q, docs, 10, chunk=7)
    expected = np.argsort(-(q @ docs.T), axis=1)[:, :10]
    assert (idx == expected).all()
    assert (np.diff(val, axis=1) <= 0).all()


def test_truncate_renormalises():
    x = normalise(np.random.default_rng(0).normal(size=(5, 32)))
    assert np.allclose(np.linalg.norm(truncate(x, 8), axis=1), 1)


def test_int8_and_binary_search_mostly_agree_with_exact_search():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(50, 64))
    docs = normalise(np.repeat(centres, 40, axis=0) + rng.normal(0, 0.3, (2000, 64)))
    q = normalise(centres[:10] + rng.normal(0, 0.3, (10, 64)))
    exact = topk(q, docs, 20)[0]
    codec = Int8Codec.fit(docs)
    int8 = codec.search(q, codec.encode(docs), 20)
    binary = binary_search(q, binary_encode(docs), 20, rescore=4)
    overlap = lambda a: np.mean([len(set(a[i]) & set(exact[i])) / 20 for i in range(10)])  # noqa: E731
    assert overlap(int8) > 0.9
    assert overlap(binary) > 0.5


def test_binary_codes_pack_eight_dims_per_byte():
    codes = binary_encode(np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1.0]]))
    assert codes.shape == (1, 2) and codes[0, 0] == 0b10101010


def test_rrf_rewards_agreement():
    a = np.array([[1, 2, 4]])
    b = np.array([[5, 2, 9]])
    fused = rrf([a, b], 3)
    assert fused[0, 0] == 2  # 2nd in both lists beats 1st in one: 2/62 > 1/61
    assert set(fused[0]) <= {1, 2, 4, 5, 9}


def test_bm25_finds_the_matching_document():
    docs = ["stock dividends and taxes", "how to bake bread", "mortgage interest rates rising"]
    ranked = BM25(docs).search(["what are dividend taxes", "the of and"], 3)
    assert ranked[0, 0] == 0
    assert ranked.shape == (2, 3)


@pytest.fixture
def mining_setup():
    rng = np.random.default_rng(0)
    docs = normalise(rng.normal(size=(300, 16)))
    queries = normalise(docs[:20] + rng.normal(0, 0.05, (20, 16)))
    positives = [{i, i + 100} for i in range(20)]
    return queries, docs, positives


def test_mined_negatives_are_never_judged_positives(mining_setup):
    queries, docs, positives = mining_setup
    triplets = mine(queries, docs, positives, MiningConfig(last_rank=10))
    assert len(triplets.query) == 40
    for q, p, n in zip(triplets.query, triplets.positive, triplets.negative):
        assert p in positives[q] and n not in positives[q]


def test_positive_aware_filter_drops_near_duplicates_of_the_positive(mining_setup):
    queries, docs, positives = mining_setup
    docs = docs.copy()
    docs[200] = docs[0]  # an unlabelled copy of query 0's positive
    triplets = mine(queries, docs, positives, MiningConfig(last_rank=10, max_ratio=0.95), chosen=[0] + [i for i in range(1, 20)])
    assert triplets.negative[0] != 200
    assert triplets.stats["filtered_candidates"] >= 1
    unfiltered = mine(queries, docs, positives, MiningConfig(last_rank=1), chosen=list(range(20)))
    assert unfiltered.negative[0] == 200


def test_pick_positives_is_deterministic():
    p = [{1, 2, 3}, {4}]
    assert pick_positives(p, 0) == pick_positives(p, 0)
    assert pick_positives(p)[1] == 4


def test_from_frames_keeps_only_positive_judgements_and_joins_titles():
    corpus = pd.DataFrame({"_id": ["d1", "d2"], "title": ["Tax", ""], "text": ["on dividends", "bread"]})
    queries = pd.DataFrame({"_id": ["q1", "q2"], "text": ["dividend tax", "unjudged"]})
    qrels = {"test": pd.DataFrame({"query-id": ["q1", "q1"], "corpus-id": ["d1", "d2"], "score": ["1", "0"]})}
    data = from_frames("toy", corpus, queries, qrels)
    assert data.doc_texts == ["Tax on dividends", "bread"]
    qids, texts, positives = data.split("test")
    assert qids == ["q1"] and texts == ["dividend tax"] and positives == [{0}]
