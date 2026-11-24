"""Production retrieval index wrapper, FastAPI serving app, and CLI entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from .config import IndexConfig, IndexRequest, RetrievalMode, SearchHit, SearchRequest, SearchResponse, ServeConfig
from .retrieval import BM25, Int8Codec, binary_encode, binary_search, normalise, rrf, topk, truncate


class RetrievalIndex:
    """In-memory retrieval index supporting BM25, dense float32, Matryoshka, int8, binary, and RRF hybrid search."""

    def __init__(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        embeddings: np.ndarray | None = None,
        config: IndexConfig | None = None,
        encoder: Callable[[list[str]], np.ndarray] | None = None,
    ) -> None:
        if not doc_ids or len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must be non-empty and have the same length")
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        self.config = config or IndexConfig()
        self.encoder = encoder
        self.bm25 = BM25(self.doc_texts)

        self.dense: np.ndarray | None = None
        self.int8_codec: Int8Codec | None = None
        self.int8_codes: np.ndarray | None = None
        self.binary_codes: np.ndarray | None = None
        self.dim: int | None = None

        if embeddings is None and encoder is not None:
            embeddings = encoder(self.doc_texts)
        if embeddings is not None:
            vecs = normalise(np.asarray(embeddings, dtype=np.float32))
            if vecs.ndim != 2 or vecs.shape[0] != len(self.doc_ids):
                raise ValueError("embeddings must be 2D with one row per document")
            if self.config.truncate_dim is not None:
                if self.config.truncate_dim > vecs.shape[1]:
                    raise ValueError("truncate_dim exceeds embedding dimension")
                vecs = truncate(vecs, self.config.truncate_dim)
            self.dense = vecs
            self.dim = int(vecs.shape[1])
            self.int8_codec = Int8Codec.fit(vecs)
            self.int8_codes = self.int8_codec.encode(vecs)
            self.binary_codes = binary_encode(vecs)

    def _query_vec(self, query: str, query_vector: list[float] | None) -> np.ndarray:
        if query_vector is not None:
            vec = normalise(np.asarray([query_vector], dtype=np.float32))
        elif self.encoder is not None:
            vec = normalise(np.asarray(self.encoder([query]), dtype=np.float32))
        else:
            raise ValueError("dense/compressed/hybrid search requires query_vector or an index encoder")
        if self.config.truncate_dim is not None:
            vec = truncate(vec, self.config.truncate_dim)
        if self.dim is not None and vec.shape[1] != self.dim:
            raise ValueError(f"query vector dim {vec.shape[1]} does not match index dim {self.dim}")
        return vec

    def search(self, query: str, top_k: int = 10, mode: RetrievalMode | None = None, query_vector: list[float] | None = None) -> list[SearchHit]:
        active_mode = mode or self.config.mode
        k = min(top_k, len(self.doc_ids))
        scores: list[float | None] = [None] * k

        if active_mode == "bm25":
            ranked = self.bm25.search([query], k)[0]
        else:
            if self.dense is None or self.int8_codec is None or self.int8_codes is None or self.binary_codes is None:
                raise ValueError(f"mode '{active_mode}' requires document embeddings in the index")
            q_vec = self._query_vec(query, query_vector)
            if active_mode == "dense":
                idx, vals = topk(q_vec, self.dense, k)
                ranked = idx[0]
                scores = [round(float(v), 6) for v in vals[0]]
            elif active_mode == "int8":
                ranked = self.int8_codec.search(q_vec, self.int8_codes, k)[0]
            elif active_mode == "binary":
                ranked = binary_search(q_vec, self.binary_codes, k, rescore=self.config.binary_rescore)[0]
            elif active_mode == "hybrid":
                depth = min(max(k * 2, 20), len(self.doc_ids))
                bm25_r = self.bm25.search([query], depth)
                dense_r = topk(q_vec, self.dense, depth)[0]
                ranked = rrf([dense_r, bm25_r], k, constant=self.config.rrf_constant)[0]
            else:
                raise ValueError(f"unsupported mode: {active_mode}")

        hits = []
        for rank, doc_idx in enumerate(ranked, start=1):
            if int(doc_idx) < 0:
                continue
            hits.append(
                SearchHit(
                    doc_id=self.doc_ids[int(doc_idx)],
                    text=self.doc_texts[int(doc_idx)],
                    rank=rank,
                    score=scores[rank - 1] if rank - 1 < len(scores) else None,
                )
            )
        return hits

    def stats(self) -> dict:
        return {
            "documents": len(self.doc_ids),
            "mode": self.config.mode,
            "dim": self.dim,
            "has_dense": self.dense is not None,
            "bytes_per_doc": {
                "float32": self.dim * 4 if self.dim else 0,
                "int8": self.dim if self.dim else 0,
                "binary": (self.dim + 7) // 8 if self.dim else 0,
            },
        }


def create_app(index: RetrievalIndex | None = None):
    """Create a FastAPI application serving retrieval queries."""
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="embedtune Retrieval API", version="0.1.0")
    state: dict[str, RetrievalIndex | None] = {"index": index}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "indexed": state["index"] is not None}

    @app.get("/stats")
    def stats() -> dict:
        if state["index"] is None:
            return {"indexed": False, "documents": 0}
        return {"indexed": True, **state["index"].stats()}

    @app.post("/index")
    def build_index(req: IndexRequest) -> dict:
        doc_ids = [d.doc_id for d in req.documents]
        doc_texts = [d.text for d in req.documents]
        embs = np.asarray([d.embedding for d in req.documents], dtype=np.float32) if req.documents[0].embedding is not None else None
        state["index"] = RetrievalIndex(doc_ids=doc_ids, doc_texts=doc_texts, embeddings=embs, config=req.config)
        return {"status": "indexed", **state["index"].stats()}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        if state["index"] is None:
            raise HTTPException(status_code=400, detail="index is empty; POST /index first")
        active_mode = req.mode or state["index"].config.mode
        try:
            hits = state["index"].search(req.query, top_k=req.top_k, mode=active_mode, query_vector=req.query_vector)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return SearchResponse(query=req.query, mode=active_mode, hits=hits)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve embedtune retrieval API via uvicorn.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", default="bm25", choices=["bm25", "dense", "int8", "binary", "hybrid"])
    parser.add_argument("--truncate-dim", type=int, default=None)
    args = parser.parse_args()

    cfg = ServeConfig(host=args.host, port=args.port, mode=args.mode, truncate_dim=args.truncate_dim)
    import uvicorn

    uvicorn.run(create_app(), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
