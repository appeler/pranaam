"""PyTorch model and text preprocessing for name classification."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path

_DELETE_ASCII_PUNCTUATION: Final[dict[int, int | None]] = str.maketrans(
    "", "", string.punctuation
)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and preprocessing constants shared by both models."""

    vocabulary_size: int = 300_000
    embedding_rows: int = 300_001
    embedding_dim: int = 64
    sequence_length: int = 50
    num_classes: int = 2


class NameTokenizer:
    """Reproduce the Keras ``TextVectorization`` inference transform."""

    def __init__(self, vocabulary: list[str], config: ModelConfig) -> None:
        """Build a tokenizer from an ordered Keras vocabulary."""
        if len(vocabulary) != config.vocabulary_size:
            raise ValueError(
                "Vocabulary has "
                f"{len(vocabulary)} entries; expected {config.vocabulary_size}"
            )
        if vocabulary[:2] != ["", "[UNK]"]:
            raise ValueError("Vocabulary must begin with padding and [UNK] tokens")

        self._token_to_id = {token: index for index, token in enumerate(vocabulary)}
        self._sequence_length = config.sequence_length

    @classmethod
    def from_file(cls, path: Path, config: ModelConfig) -> NameTokenizer:
        """Load a UTF-8 vocabulary file with one token per line."""
        vocabulary = path.read_text(encoding="utf-8").splitlines()
        return cls(vocabulary, config)

    def encode(self, names: list[str]) -> Tensor:
        """Normalize, index, truncate, and pad a batch of names."""
        rows: list[list[int]] = []
        for name in names:
            normalized = name.lower().translate(_DELETE_ASCII_PUNCTUATION)
            token_ids = [
                self._token_to_id.get(token, 1) for token in normalized.split()
            ][: self._sequence_length]
            token_ids.extend([0] * (self._sequence_length - len(token_ids)))
            rows.append(token_ids)
        return torch.tensor(rows, dtype=torch.long)


class NameClassifier(nn.Module):
    """Embedding-average classifier equivalent to the released Keras model."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the fixed architecture used by the English and Hindi models."""
        super().__init__()
        self.embedding = nn.Embedding(config.embedding_rows, config.embedding_dim)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return unnormalized class scores for an indexed name batch."""
        embedded = self.embedding(token_ids)
        return self.classifier(embedded.mean(dim=1))

    def predict_proba(self, token_ids: Tensor) -> Tensor:
        """Return class probabilities in inference mode."""
        with torch.inference_mode():
            return torch.softmax(self(token_ids), dim=1)


def load_classifier(path: Path, config: ModelConfig) -> NameClassifier:
    """Load a classifier from a safe, non-executable tensor file."""
    model = NameClassifier(config)
    state = load_file(path, device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
