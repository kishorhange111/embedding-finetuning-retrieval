"""Typed configuration and request/response schemas for retrieval serving."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RetrievalMode = Literal["bm25", "dense", "int8", "binary", "hybrid"]


class IndexConfig(BaseModel):
    """Configuration for building and querying a compressed or hybrid retrieval index."""

    model_config = ConfigDict(extra="forbid")

    mode: RetrievalMode = "hybrid"
    truncate_dim: int | None = Field(default=None, ge=8)
    rrf_constant: int = Field(default=60, ge=1, le=1000)
    binary_rescore: int = Field(default=4, ge=1, le=32)


class DocumentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    embedding: list[float] | None = None


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[DocumentInput] = Field(min_length=1)
    config: IndexConfig = Field(default_factory=IndexConfig)

    @model_validator(mode="after")
    def _validate_embeddings(self) -> IndexRequest:
        has_emb = [d.embedding is not None for d in self.documents]
        if any(has_emb) and not all(has_emb):
            raise ValueError("either all documents or no documents must provide embeddings")
        if all(has_emb):
            dims = {len(d.embedding) for d in self.documents if d.embedding is not None}
            if len(dims) != 1 or 0 in dims:
                raise ValueError("all document embeddings must share the same positive dimension")
            dim = next(iter(dims))
            if self.config.truncate_dim is not None and self.config.truncate_dim > dim:
                raise ValueError(f"truncate_dim ({self.config.truncate_dim}) exceeds embedding dimension ({dim})")
        return self


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=500)
    mode: RetrievalMode | None = None
    query_vector: list[float] | None = None


class SearchHit(BaseModel):
    doc_id: str
    text: str
    rank: int
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: RetrievalMode
    hits: list[SearchHit]


class ServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    mode: RetrievalMode = "bm25"
    truncate_dim: int | None = Field(default=None, ge=8)
