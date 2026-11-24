"""A training smoke test on a tiny randomly initialised BERT, so it runs offline in seconds."""

import pytest

pytest.importorskip("sentence_transformers")

WORDS = [
    "tax",
    "stock",
    "bond",
    "loan",
    "rate",
    "bank",
    "fund",
    "price",
    "save",
    "pay",
    "bread",
    "cake",
    "oven",
    "flour",
    "sugar",
    "salt",
    "milk",
    "egg",
    "rice",
    "tea",
]


@pytest.fixture
def tiny_model(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    # A dict, not vocab_file: Transformers 5 ignores vocab_file and every word would tokenise to [UNK].
    tokenizer = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)})
    assert tokenizer.convert_ids_to_tokens(tokenizer("tax bread")["input_ids"]) == ["[CLS]", "tax", "bread", "[SEP]"]
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    path = tmp_path / "tiny"
    BertModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def accuracy(model, anchors, targets):
    from embedtune.retrieval import encode

    picked = (encode(model, anchors) @ encode(model, targets).T).argmax(axis=1)
    return float((picked == range(len(anchors))).mean())


def test_finetune_learns_an_association_with_no_word_overlap(tiny_model, tmp_path):
    from sentence_transformers import SentenceTransformer

    from embedtune.train import TrainConfig, finetune

    anchors, targets = WORDS[:10], WORDS[10:]
    before = accuracy(SentenceTransformer(tiny_model, device="cpu"), anchors, targets)
    info = finetune(tiny_model, anchors * 20, targets * 20, None, TrainConfig(epochs=4, batch_size=8, lr=3e-3, max_seq_length=8), tmp_path / "out")
    assert info["steps"] > 0 and (tmp_path / "out" / "config.json").exists()
    after = accuracy(SentenceTransformer(str(tmp_path / "out"), device="cpu"), anchors, targets)
    assert after >= 0.9 > before
