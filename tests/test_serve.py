import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from embedtune.config import IndexConfig, IndexRequest, SearchRequest, ServeConfig
from embedtune.serve import RetrievalIndex, create_app


def _sample_corpus():
    doc_ids = ["d1", "d2", "d3"]
    doc_texts = [
        "stock dividends and capital gains taxes",
        "sourdough bread baking recipe with yeast",
        "mortgage interest rates and bond yields",
    ]
    embeddings = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.1, 0.0, 0.1, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return doc_ids, doc_texts, embeddings


def test_config_validation_rejects_invalid_values():
    with pytest.raises(ValidationError):
        IndexConfig(mode="invalid")
    with pytest.raises(ValidationError):
        SearchRequest(query="", top_k=5)
    with pytest.raises(ValidationError):
        SearchRequest(query="taxes", top_k=0)
    with pytest.raises(ValidationError):
        ServeConfig(port=70000)


def test_index_request_rejects_mixed_or_oversized_truncate_dim():
    with pytest.raises(ValidationError):
        IndexRequest(
            documents=[
                {"doc_id": "d1", "text": "hello", "embedding": [1.0, 0.0]},
                {"doc_id": "d2", "text": "world", "embedding": None},
            ]
        )
    with pytest.raises(ValidationError):
        IndexRequest(
            documents=[{"doc_id": "d1", "text": "hello", "embedding": [1.0] * 8}],
            config={"mode": "dense", "truncate_dim": 16},
        )


@pytest.mark.parametrize("mode", ["bm25", "dense", "int8", "binary", "hybrid"])
def test_retrieval_index_supports_all_modes(mode):
    doc_ids, doc_texts, embeddings = _sample_corpus()
    index = RetrievalIndex(doc_ids, doc_texts, embeddings=embeddings, config=IndexConfig(mode=mode))
    hits = index.search("dividend taxes", top_k=2, query_vector=embeddings[0].tolist())
    assert len(hits) == 2
    assert hits[0].doc_id == "d1"
    assert hits[0].rank == 1


def test_fastapi_endpoints_index_search_and_stats():
    doc_ids, doc_texts, embeddings = _sample_corpus()
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok", "indexed": False}
    assert client.post("/search", json={"query": "taxes"}).status_code == 400

    payload = {
        "documents": [
            {"doc_id": i, "text": t, "embedding": e.tolist()}
            for i, t, e in zip(doc_ids, doc_texts, embeddings)
        ],
        "config": {"mode": "hybrid", "rrf_constant": 60},
    }
    res = client.post("/index", json=payload)
    assert res.status_code == 200
    assert res.json()["documents"] == 3
    assert res.json()["bytes_per_doc"]["float32"] == 32

    search_res = client.post(
        "/search",
        json={"query": "dividend taxes", "top_k": 2, "query_vector": embeddings[0].tolist()},
    )
    assert search_res.status_code == 200
    body = search_res.json()
    assert body["mode"] == "hybrid"
    assert body["hits"][0]["doc_id"] == "d1"

    bad_dim = client.post("/search", json={"query": "taxes", "mode": "dense", "query_vector": [1.0, 2.0]})
    assert bad_dim.status_code == 422

    stats = client.get("/stats").json()
    assert stats["indexed"] is True and stats["dim"] == 8
