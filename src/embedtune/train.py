"""Contrastive fine-tuning with in-batch negatives (MultipleNegativesRankingLoss), optionally with mined hard negatives
and a Matryoshka objective."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    batch_size: int = 64
    lr: float = 5e-5
    warmup_ratio: float = 0.1
    # Training truncates documents harder than evaluation (256) to fit a CPU budget; most FiQA answers fit anyway.
    max_seq_length: int = 128
    matryoshka_dims: tuple[int, ...] | None = None
    seed: int = 0


def finetune(model_name_or_path: str, anchors: list[str], positives: list[str], negatives: list[str] | None, cfg: TrainConfig, out_dir: Path) -> dict:
    """Train and save to ``out_dir``. Returns timing and the loss trace."""
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    model = SentenceTransformer(model_name_or_path, device="cpu")
    model.max_seq_length = cfg.max_seq_length
    columns = {"anchor": anchors, "positive": positives}
    if negatives is not None:
        columns["negative"] = negatives
    train = Dataset.from_dict(columns)
    loss = MultipleNegativesRankingLoss(model)
    if cfg.matryoshka_dims:
        loss = MatryoshkaLoss(model, loss, matryoshka_dims=list(cfg.matryoshka_dims))
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_ratio,  # Transformers 5 reads a float below 1 as a fraction of all steps
        # Several judged positives share a query; without this, one of them is scored as the other's negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=cfg.seed,
        data_seed=cfg.seed,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss)
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started
    model.save(str(out_dir))
    losses = [round(e["loss"], 4) for e in trainer.state.log_history if "loss" in e]
    return {"train_seconds": round(seconds, 1), "steps": trainer.state.global_step, "rows": len(train), "loss_trace": losses, "config": asdict(cfg)}
