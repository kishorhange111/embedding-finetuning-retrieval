# Model Card & Reproducibility Report — `embedtune`

## 1. Model & System Overview

- **System**: `embedtune` — Domain-adapted dense retrieval pipeline with positive-aware hard-negative mining, Matryoshka Representation Learning (MRL), scalar `int8` & 1-bit binary quantization, and BM25+dense Reciprocal Rank Fusion (RRF).
- **Base Encoder**: `sentence-transformers/all-MiniLM-L6-v2` (22M parameters, 384-dim embeddings, 256-token evaluation sequence length).
- **Lexical Baseline**: `bm25s` with English stopwords and Snowball stemming.
- **Intended Use**: Domain retrieval benchmarking, compressed vector index evaluation, and low-footprint hybrid search serving via `embedtune-serve`.

## 2. Datasets & Splits

- **In-Domain Target**: `BeIR/fiqa` (Financial Opinion Mining & QA; `train`, `dev`, `test` splits). Model selection is strictly restricted to `fiqa/dev` `nDCG@10`; `fiqa/test` is scored once for reporting.
- **Out-of-Domain Transfer**: `BeIR/scifact` (`test` split) — never trained on; measures domain-adaptation forgetting.

## 3. Training & Compression Protocol

Pre-registered variants in `embedtune.evaluate.VARIANTS`:

| Variant | Negative Mining | Matryoshka Dims | Loss & Batching |
| :--- | :--- | :--- | :--- |
| `in_batch` | None | Full 384d | `MultipleNegativesRankingLoss` (`NO_DUPLICATES`) |
| `hard_top` | Top ranks `[0, 10)`, unfiltered | Full 384d | `MultipleNegativesRankingLoss` |
| `hard_filtered` | Ranks `[0, 50)`, `score < 0.95 * pos` | Full 384d | `MultipleNegativesRankingLoss` |
| `hard_filtered_matryoshka` | Ranks `[0, 50)`, `score < 0.95 * pos` | `(384, 192, 96, 48)` | `MatryoshkaLoss(MNRL)` |

Training defaults (`TrainConfig`): `epochs=1`, `batch_size=64`, `lr=5e-5`, `warmup_ratio=0.1`, `max_seq_length=128`, `seed=0`.

## 4. Observed Reproducible Benchmarks

From `eval.log` (`python -m embedtune.evaluate`, depth = 100):

| System | Dataset / Split | nDCG@10 | Recall@100 |
| :--- | :--- | :---: | :---: |
| `bm25` | `fiqa/dev` | 0.2592 | 0.5495 |
| `bm25` | `fiqa/test` | 0.2514 | 0.5593 |
| `bm25` | `scifact/test` | 0.6863 | 0.9127 |

### Deterministic Index Memory Footprint

| Compression Mode | Bytes / Document | Reduction vs `float32_384` |
| :--- | :---: | :---: |
| `float32_384` | 1,536 | 1x |
| `float32_192` | 768 | 2x |
| `int8_384` | 384 | 4x |
| `float32_96` | 384 | 4x |
| `float32_48` | 192 | 8x |
| `binary_384` | 48 | 32x |

## 5. Reproducibility Instructions

```bash
# 1. Run deterministic unit & serving tests (offline)
pytest -v

# 2. Reproduce full BEIR FiQA + SciFact evaluation (requires HF Hub download)
embedtune-eval --out results --threads 4

# 3. Build & run serving container
docker build -t embedtune:latest .
docker run -p 8000:8000 embedtune:latest
```

## 6. Limitations & Risks

- **Incomplete Judgements**: BEIR qrels are sparse; even with `max_ratio=0.95` filtering, some mined negatives may still be unlabelled positives.
- **Truncation Length**: Training uses `max_seq_length=128` for CPU tractability while evaluation uses `256`; very long financial filings require chunking before indexing.
- **Quantization Calibration**: `Int8Codec` calibrates per-dimension min/max bounds on a corpus sample; severe domain shift in new documents can clip extreme coordinates.
