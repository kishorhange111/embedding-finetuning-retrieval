"""BEIR datasets from the Hugging Face Hub: corpus, queries and relevance judgements per split."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Beir:
    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    queries: dict[str, str]
    # split -> query id -> doc id -> graded relevance (only judgements above 0 are kept)
    qrels: dict[str, dict[str, dict[str, int]]]
    doc_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.doc_index = {doc: i for i, doc in enumerate(self.doc_ids)}

    def split(self, name: str) -> tuple[list[str], list[str], list[set[int]]]:
        """Query ids, query texts and the indices of each query's relevant documents."""
        judged = self.qrels[name]
        qids = [q for q in judged if q in self.queries]
        positives = [{self.doc_index[d] for d in judged[q] if d in self.doc_index} for q in qids]
        keep = [i for i, p in enumerate(positives) if p]
        return [qids[i] for i in keep], [self.queries[qids[i]] for i in keep], [positives[i] for i in keep]


def doc_text(title: str | None, text: str | None) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    return f"{title} {text}".strip() if title else text


def from_frames(name: str, corpus: pd.DataFrame, queries: pd.DataFrame, qrels: dict[str, pd.DataFrame]) -> Beir:
    """Build from BEIR-format tables: corpus (_id, title, text), queries (_id, text), qrels (query-id, corpus-id, score)."""
    titles = corpus["title"] if "title" in corpus else [""] * len(corpus)
    texts = [doc_text(t, x) for t, x in zip(titles, corpus["text"])]
    judged: dict[str, dict[str, dict[str, int]]] = {}
    for split, frame in qrels.items():
        table: dict[str, dict[str, int]] = {}
        for q, d, s in zip(frame["query-id"].astype(str), frame["corpus-id"].astype(str), frame["score"].astype(int)):
            if s > 0:
                table.setdefault(q, {})[d] = s
        judged[split] = table
    return Beir(
        name=name,
        doc_ids=corpus["_id"].astype(str).tolist(),
        doc_texts=texts,
        queries=dict(zip(queries["_id"].astype(str), queries["text"].astype(str))),
        qrels=judged,
    )


def load(name: str, splits: tuple[str, ...] = ("train", "dev", "test")) -> Beir:
    """Download ``BeIR/<name>`` and ``BeIR/<name>-qrels``. Splits a dataset doesn't publish are skipped."""
    from huggingface_hub import hf_hub_download, list_repo_files

    def parquet(folder: str) -> pd.DataFrame:
        files = [f for f in list_repo_files(f"BeIR/{name}", repo_type="dataset") if f.startswith(f"{folder}/") and f.endswith(".parquet")]
        return pd.concat([pd.read_parquet(hf_hub_download(f"BeIR/{name}", f, repo_type="dataset")) for f in sorted(files)])

    available = set(list_repo_files(f"BeIR/{name}-qrels", repo_type="dataset"))
    qrels = {
        split: pd.read_csv(hf_hub_download(f"BeIR/{name}-qrels", f"{split}.tsv", repo_type="dataset"), sep="\t", dtype=str)
        for split in splits
        if f"{split}.tsv" in available
    }
    return from_frames(name, parquet("corpus"), parquet("queries"), qrels)
