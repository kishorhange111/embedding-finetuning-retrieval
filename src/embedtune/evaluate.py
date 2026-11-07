"""Run every comparison and write ``summary.json``.

    embedtune-eval --out results

Configurations are compared on FiQA's dev queries; the test queries are scored for all of them, but the chosen
configuration is the one with the best dev nDCG@10. SciFact (a different domain, never trained on) measures what
fine-tuning costs elsewhere.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data import Beir, load
from .metrics import paired_diff, per_query, summarise
from .mining import MiningConfig, mine, pick_positives
from .retrieval import BM25, Int8Codec, binary_encode, binary_search, encode, rrf, topk, truncate
from .train import TrainConfig, finetune

BASE = "sentence-transformers/all-MiniLM-L6-v2"
EVAL_MAX_SEQ_LENGTH = 256
DEPTH = 100

# name -> (negatives, matryoshka). Chosen before any run and not extended after reading test numbers.
VARIANTS: dict[str, dict] = {
    "in_batch": {"mining": None, "matryoshka": False},
    "hard_filtered": {"mining": MiningConfig(first_rank=0, last_rank=50, max_ratio=0.95), "matryoshka": False},
    "hard_top": {"mining": MiningConfig(first_rank=0, last_rank=10), "matryoshka": False},
    "hard_filtered_matryoshka": {"mining": MiningConfig(first_rank=0, last_rank=50, max_ratio=0.95), "matryoshka": True},
}
MATRYOSHKA_DIMS = (384, 192, 96, 48)


class Encoded:
    """A model's embeddings of a dataset's corpus, cached on disk because the corpus is the slow part."""

    def __init__(self, model, data: Beir, cache: Path, tag: str):
        path = cache / f"{tag}-{data.name}-corpus.npy"
        started = time.perf_counter()
        if path.exists():
            self.docs = np.load(path)
            self.corpus_seconds = None
        else:
            self.docs = encode(model, data.doc_texts)
            self.corpus_seconds = round(time.perf_counter() - started, 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.docs)
        self.model = model
        self.data = data

    def queries(self, split: str) -> tuple[np.ndarray, list[set[int]]]:
        _, texts, positives = self.data.split(split)
        return encode(self.model, texts), positives


def load_model(path: str):
    """Every model is evaluated at the base model's 256-token limit, whatever length it was trained at."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(path, device="cpu")
    model.max_seq_length = EVAL_MAX_SEQ_LENGTH
    return model


def score(ranked: np.ndarray, positives: list[set[int]]) -> dict[str, np.ndarray]:
    return per_query(ranked, positives)


def dense_runs(enc: Encoded, splits: list[str]) -> dict[str, dict]:
    out = {}
    for split in splits:
        q, positives = enc.queries(split)
        out[split] = {"ranked": topk(q, enc.docs, DEPTH)[0], "positives": positives, "q": q}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="results")
    parser.add_argument("--models", default="models")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    out, models, cache = Path(args.out), Path(args.models), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"base_model": BASE, "depth": DEPTH}
    per_query_scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    def record(name: str, dataset: str, split: str, values: dict[str, np.ndarray]) -> None:
        per_query_scores.setdefault(name, {})[f"{dataset}/{split}"] = values
        report.setdefault("systems", {}).setdefault(name, {})[f"{dataset}/{split}"] = summarise(values)
        print(f"{name:28s} {dataset}/{split:5s} nDCG@10 {values['ndcg@10'].mean():.4f}  R@100 {values['recall@100'].mean():.4f}", flush=True)

    def save() -> None:
        (out / "summary.json").write_text(json.dumps(report, indent=2))

    fiqa = load("fiqa")
    scifact = load("scifact", splits=("test",))
    report["data"] = {
        "fiqa": {"docs": len(fiqa.doc_ids), **{s: len(fiqa.split(s)[0]) for s in fiqa.qrels}},
        "scifact": {"docs": len(scifact.doc_ids), "test": len(scifact.split("test")[0])},
    }

    # Lexical baseline.
    for data, splits in ((fiqa, ["dev", "test"]), (scifact, ["test"])):
        bm25 = BM25(data.doc_texts)
        for split in splits:
            _, texts, positives = data.split(split)
            record("bm25", data.name, split, score(bm25.search(texts, DEPTH), positives))

    # Base model.
    base = load_model(BASE)
    base_fiqa = Encoded(base, fiqa, cache, "base")
    report["base_corpus_encode_seconds"] = base_fiqa.corpus_seconds
    runs = {"base": dense_runs(base_fiqa, ["dev", "test"])}
    for split, run in runs["base"].items():
        record("base", "fiqa", split, score(run["ranked"], run["positives"]))
    base_sci = Encoded(base, scifact, cache, "base")
    runs_sci = {"base": dense_runs(base_sci, ["test"])}
    record("base", "scifact", "test", score(runs_sci["base"]["test"]["ranked"], runs_sci["base"]["test"]["positives"]))
    save()

    # Training pairs, with negatives mined by the base model.
    _, train_texts, train_positives = fiqa.split("train")
    train_q = encode(base, train_texts)
    chosen = pick_positives(train_positives)
    report["training"] = {}
    for name in args.variants.split(","):
        spec = VARIANTS[name]
        anchors, positives, negatives = list(train_texts), [fiqa.doc_texts[p] for p in chosen], None
        mining_stats = None
        if spec["mining"] is not None:
            triplets = mine(train_q, base_fiqa.docs, train_positives, spec["mining"], chosen)
            anchors = [train_texts[i] for i in triplets.query]
            positives = [fiqa.doc_texts[i] for i in triplets.positive]
            negatives = [fiqa.doc_texts[i] for i in triplets.negative]
            mining_stats = {**triplets.stats, "config": asdict(spec["mining"])}
        cfg = TrainConfig(matryoshka_dims=MATRYOSHKA_DIMS if spec["matryoshka"] else None)
        model_dir = models / name
        if (model_dir / "config.json").exists():
            info = json.loads((model_dir / "train_info.json").read_text())
        else:
            info = finetune(BASE, anchors, positives, negatives, cfg, model_dir)
            (model_dir / "train_info.json").write_text(json.dumps(info, indent=2))
        report["training"][name] = {**info, "mining": mining_stats}
        print(f"trained {name}: {info['steps']} steps in {info['train_seconds']} s", flush=True)

        model = load_model(str(model_dir))
        enc = Encoded(model, fiqa, cache, name)
        runs[name] = dense_runs(enc, ["dev", "test"])
        for split, run in runs[name].items():
            record(name, "fiqa", split, score(run["ranked"], run["positives"]))
        enc_sci = Encoded(model, scifact, cache, name)
        runs_sci[name] = dense_runs(enc_sci, ["test"])
        record(name, "scifact", "test", score(runs_sci[name]["test"]["ranked"], runs_sci[name]["test"]["positives"]))
        save()

    # Pick on dev only.
    tuned = [n for n in runs if n != "base"]
    best = max(tuned, key=lambda n: report["systems"][n]["fiqa/dev"]["ndcg@10"]["mean"])
    report["selected_on_dev"] = best

    # Hybrid with BM25 (fixed RRF constant 60, no tuning).
    bm25 = BM25(fiqa.doc_texts)
    for name in ("base", best):
        for split in ("dev", "test"):
            run = runs[name][split]
            _, texts, _ = fiqa.split(split)
            fused = rrf([run["ranked"], bm25.search(texts, DEPTH)], DEPTH)
            record(f"{name}+bm25_rrf", "fiqa", split, score(fused, run["positives"]))

    # Paired differences on test.
    test = {n: s["fiqa/test"] for n, s in per_query_scores.items() if "fiqa/test" in s}
    comparisons = [(best, "base"), (best, "bm25"), ("base", "bm25"), (f"{best}+bm25_rrf", best)]
    comparisons += [(n, "in_batch") for n in tuned if n != "in_batch"]
    report["paired_test"] = {
        f"{a} - {b}": {m: paired_diff(test[a][m], test[b][m]) for m in ("ndcg@10", "recall@100")} for a, b in comparisons if a in test and b in test
    }
    sci = {n: s["scifact/test"] for n, s in per_query_scores.items() if "scifact/test" in s}
    report["paired_scifact"] = {f"{n} - base": paired_diff(sci[n]["ndcg@10"], sci["base"]["ndcg@10"]) for n in tuned}
    save()

    # Compression on FiQA test: truncated dimensions and quantised codes, with and without Matryoshka training.
    compression: dict = {}
    for name in ("base", best, "hard_filtered_matryoshka"):
        if name not in runs:
            continue
        docs = np.load(cache / f"{name}-fiqa-corpus.npy")
        q = runs[name]["test"]["q"]
        positives = runs[name]["test"]["positives"]
        rows = {}
        for dim in MATRYOSHKA_DIMS:
            ranked = topk(truncate(q, dim), truncate(docs, dim), DEPTH)[0]
            rows[f"float32_{dim}"] = {"bytes_per_doc": dim * 4, **{m: round(float(v.mean()), 4) for m, v in score(ranked, positives).items()}}
        sample = docs[np.random.default_rng(0).choice(len(docs), 5000, replace=False)]
        codec = Int8Codec.fit(sample)
        ranked = codec.search(q, codec.encode(docs), DEPTH)
        rows["int8_384"] = {"bytes_per_doc": 384, **{m: round(float(v.mean()), 4) for m, v in score(ranked, positives).items()}}
        codes = binary_encode(docs)
        for rescore in (1, 4):
            ranked = binary_search(q, codes, DEPTH, rescore=rescore)
            rows[f"binary_384_rescore{rescore}"] = {"bytes_per_doc": 48, **{m: round(float(v.mean()), 4) for m, v in score(ranked, positives).items()}}
        compression[name] = rows
        print(name, json.dumps(rows), flush=True)
    report["compression_fiqa_test"] = compression
    save()

    # Query latency of the selected model (one query at a time, exact search over the corpus).
    model = load_model(str(models / best))
    docs = np.load(cache / f"{best}-fiqa-corpus.npy")
    _, texts, _ = fiqa.split("test")
    timings = []
    for text in texts[:200]:
        started = time.perf_counter()
        topk(encode(model, [text]), docs, 10)
        timings.append((time.perf_counter() - started) * 1000)
    report["query_latency_ms"] = {"p50": round(float(np.percentile(timings, 50)), 2), "p99": round(float(np.percentile(timings, 99)), 2), "threads": args.threads}
    save()
    print(json.dumps({k: report[k] for k in ("selected_on_dev", "paired_test", "query_latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
