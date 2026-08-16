"""Tests for the PyTorch architecture and Keras-compatible tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file

from pranaam.model import ModelConfig, NameClassifier, NameTokenizer, load_classifier

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def small_config() -> ModelConfig:
    """Return a cheap architecture with production-equivalent behavior."""
    return ModelConfig(
        vocabulary_size=4,
        embedding_rows=5,
        embedding_dim=2,
        sequence_length=3,
        num_classes=2,
    )


def test_tokenizer_matches_normalization_and_padding(
    small_config: ModelConfig,
) -> None:
    """Case and ASCII punctuation are removed before indexing and padding."""
    tokenizer = NameTokenizer(["", "[UNK]", "foo", "bar"], small_config)

    encoded = tokenizer.encode(["FOO, bar!", "unknown", "foo bar foo bar"])

    assert encoded.tolist() == [[2, 3, 0], [1, 0, 0], [2, 3, 2]]


def test_tokenizer_rejects_wrong_vocabulary_size(small_config: ModelConfig) -> None:
    """Incomplete vocabularies fail before inference."""
    with pytest.raises(ValueError, match="expected 4"):
        NameTokenizer(["", "[UNK]"], small_config)


def test_tokenizer_rejects_missing_reserved_tokens(
    small_config: ModelConfig,
) -> None:
    """Padding and unknown IDs must retain their Keras positions."""
    with pytest.raises(ValueError, match=r"padding and \[UNK\]"):
        NameTokenizer(["bad", "tokens", "foo", "bar"], small_config)


def test_tokenizer_loads_utf8_vocabulary(
    small_config: ModelConfig, tmp_path: Path
) -> None:
    """Hindi tokens round-trip through the external vocabulary file."""
    path = tmp_path / "vocabulary.txt"
    path.write_text("\n[UNK]\nराम\nखान\n", encoding="utf-8")

    tokenizer = NameTokenizer.from_file(path, small_config)

    assert tokenizer.encode(["राम खान"]).tolist() == [[2, 3, 0]]


def test_classifier_includes_padding_in_mean(small_config: ModelConfig) -> None:
    """The port preserves Keras's learned padding-row contribution."""
    model = NameClassifier(small_config)
    with torch.no_grad():
        model.embedding.weight.zero_()
        model.embedding.weight[0] = torch.tensor([3.0, 6.0])
        model.embedding.weight[2] = torch.tensor([0.0, 3.0])
        model.classifier.weight.copy_(torch.eye(2))
        model.classifier.bias.zero_()

    logits = model(torch.tensor([[2, 0, 0]]))

    assert torch.allclose(logits, torch.tensor([[2.0, 5.0]]))
    probability_sum = model.predict_proba(torch.tensor([[2, 0, 0]])).sum()
    assert torch.allclose(probability_sum, torch.tensor(1.0))


def test_load_classifier_restores_safe_state(
    small_config: ModelConfig, tmp_path: Path
) -> None:
    """Safetensors restores the complete architecture in evaluation mode."""
    source = NameClassifier(small_config)
    path = tmp_path / "model.safetensors"
    save_file(source.state_dict(), path)

    loaded = load_classifier(path, small_config)

    assert loaded.training is False
    for name, tensor in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
