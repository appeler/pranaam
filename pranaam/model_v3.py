"""Byte-level PyTorch model for name-pattern estimation."""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

if TYPE_CHECKING:
    from pathlib import Path

_PAD_ID: Final[int] = 0
_START_ID: Final[int] = 1
_END_ID: Final[int] = 2
_BYTE_OFFSET: Final[int] = 3
_BYTE_VOCABULARY_SIZE: Final[int] = 256 + _BYTE_OFFSET
CURRENT_METADATA_SCHEMA_VERSION: Final[int] = 2
_PINNED_V3_METADATA_SCHEMA_VERSION: Final[int] = 1


class ReferencePopulation(StrEnum):
    """Population against which an estimate's score is calibrated."""

    SEPRI_HOUSEHOLD_HEADS = "SEPRI household heads"
    BIHAR_LAND_RECORD_NAMES = "Bihar land-record names"


class LabelSource(StrEnum):
    """Observed variables used to construct binary training labels."""

    LAND_CASTE_AND_SEPRI_RELIGION = (
        "Bihar land caste/community labels and SEPRI household religion"
    )
    LAND_CASTE = "Bihar land caste/community labels"


@dataclass(frozen=True)
class ByteModelConfig:
    """Architecture and tokenizer settings for the v3 model."""

    max_bytes: int = 128
    embedding_dim: int = 32
    hidden_dim: int = 64
    output_dim: int = 96
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Reject architecture settings that cannot produce a valid model."""
        if self.max_bytes < 3:
            raise ValueError("max_bytes must leave room for boundaries and content")
        dimensions = (self.embedding_dim, self.hidden_dim, self.output_dim)
        if any(dimension < 1 for dimension in dimensions):
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")


def normalize_name(name: str) -> str:
    """Return the canonical string consumed by the byte tokenizer."""
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def supports_name_script(name: str, supported_scripts: tuple[str, ...]) -> bool:
    """Return whether every letter belongs to a script supported by the model."""
    saw_letter = False
    for character in unicodedata.normalize("NFKC", name):
        if not unicodedata.category(character).startswith("L"):
            continue
        saw_letter = True
        unicode_name = unicodedata.name(character, "")
        if not any(script in unicode_name for script in supported_scripts):
            return False
    return saw_letter


class ByteTokenizer:
    """Encode normalized UTF-8 bytes with explicit boundaries and padding."""

    def __init__(self, max_bytes: int) -> None:
        """Create a tokenizer with a fixed encoded sequence length."""
        if max_bytes < 3:
            raise ValueError("max_bytes must leave room for boundaries and content")
        self.max_bytes = max_bytes

    def encode(self, names: list[str]) -> Tensor:
        """Encode a batch of names as fixed-length byte-ID tensors."""
        rows: list[list[int]] = []
        content_limit = self.max_bytes - 2
        for name in names:
            content = normalize_name(name).encode("utf-8")[:content_limit]
            token_ids = [_START_ID, *(byte + _BYTE_OFFSET for byte in content), _END_ID]
            token_ids.extend([_PAD_ID] * (self.max_bytes - len(token_ids)))
            rows.append(token_ids)
        return torch.tensor(rows, dtype=torch.long)

    def normalized_utf8_length(self, name: str) -> int:
        """Return the byte length used to determine tokenizer support."""
        return len(normalize_name(name).encode("utf-8"))

    def truncates(self, name: str) -> bool:
        """Return whether encoding drops normalized UTF-8 content bytes."""
        return self.normalized_utf8_length(name) > self.max_content_bytes

    @property
    def max_content_bytes(self) -> int:
        """Return the UTF-8 content capacity after boundary tokens."""
        return self.max_bytes - 2


class ByteNameClassifier(nn.Module):
    """Compact convolutional classifier that retains local byte order."""

    def __init__(self, config: ByteModelConfig) -> None:
        """Create the v3 byte-level architecture."""
        super().__init__()
        self.embedding = nn.Embedding(
            _BYTE_VOCABULARY_SIZE,
            config.embedding_dim,
            padding_idx=_PAD_ID,
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(
                config.embedding_dim,
                config.hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            for kernel_size in (3, 5, 7, 9)
        )
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim * len(self.convolutions), config.output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.classifier = nn.Linear(config.output_dim, 1)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return one uncalibrated Muslim-pattern logit per input name."""
        embedded = self.embedding(token_ids).transpose(1, 2)
        features = torch.cat(
            [
                torch.amax(torch.nn.functional.gelu(layer(embedded)), dim=2)
                for layer in self.convolutions
            ],
            dim=1,
        )
        return self.classifier(self.projection(features)).squeeze(1)

    def predict_proba(self, token_ids: Tensor) -> Tensor:
        """Return uncalibrated two-class softmax-equivalent probabilities."""
        with torch.inference_mode():
            muslim = torch.sigmoid(self(token_ids))
            return torch.stack((1 - muslim, muslim), dim=1)

    def predict_logits(self, token_ids: Tensor) -> Tensor:
        """Return uncalibrated logits in inference mode."""
        with torch.inference_mode():
            return self(token_ids)


@dataclass(frozen=True)
class CalibrationConfig:
    """Platt-scaling parameters estimated on a held-out calibration set."""

    slope: float
    intercept: float
    population: ReferencePopulation
    rows: int

    def __post_init__(self) -> None:
        """Reject invalid or non-identifiable calibration parameters."""
        if not math.isfinite(self.slope) or self.slope <= 0:
            raise ValueError("calibration slope must be finite and positive")
        if not math.isfinite(self.intercept):
            raise ValueError("calibration intercept must be finite")
        if self.rows < 1:
            raise ValueError("calibration rows must be positive")


@dataclass(frozen=True)
class ModelProvenance:
    """Typed population, labeling, and training lineage for one model."""

    reference_population: ReferencePopulation
    label_source: LabelSource
    training_seed: int
    normalization: str

    def __post_init__(self) -> None:
        """Reject incomplete training provenance."""
        if self.training_seed < 0:
            raise ValueError("training_seed cannot be negative")
        if not self.normalization:
            raise ValueError("normalization cannot be empty")


def _reference_prior(document: dict[str, object]) -> float | None:
    """Return the positive-class share of the calibrated evaluation split.

    This is the base rate the shipped calibration is anchored to, and the
    reference point a caller shifts away from with ``prior``. Returns None
    when the artifact reports no usable confusion matrix.
    """
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        return None
    calibrated = evaluation.get("calibrated")
    if not isinstance(calibrated, dict):
        return None
    try:
        positives = int(calibrated["true_positive"]) + int(calibrated["false_negative"])
        rows = int(calibrated["rows"])
    except (KeyError, TypeError, ValueError):
        return None
    if rows <= 0 or not 0 < positives < rows:
        return None
    return positives / rows


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Validated inference metadata stored beside each v3 model."""

    schema_version: int
    model_version: str
    language: str
    supported_scripts: tuple[str, ...]
    confidence_threshold: float
    architecture: ByteModelConfig
    calibration: CalibrationConfig
    provenance: ModelProvenance
    reference_prior: float | None = None

    @classmethod
    def from_file(cls, path: Path) -> ModelArtifactMetadata:
        """Read and validate a v3 model metadata document."""
        document = json.loads(path.read_text(encoding="utf-8"))
        schema_version = document.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version
            not in {
                _PINNED_V3_METADATA_SCHEMA_VERSION,
                CURRENT_METADATA_SCHEMA_VERSION,
            }
        ):
            raise ValueError("Unsupported model metadata schema")
        threshold = float(document["abstention"]["confidence_threshold"])
        if not 0.5 < threshold < 1:
            raise ValueError("confidence_threshold must be between 0.5 and 1")
        scripts = tuple(str(value) for value in document["supported_scripts"])
        if not scripts:
            raise ValueError("supported_scripts cannot be empty")
        language = str(document["language"])
        if language not in {"eng", "hin"}:
            raise ValueError("model metadata language must be 'eng' or 'hin'")
        model_version = str(document["model_version"])
        if not model_version:
            raise ValueError("model_version cannot be empty")
        calibration_document = document["calibration"]
        if schema_version == _PINNED_V3_METADATA_SCHEMA_VERSION:
            provenance_document = _pinned_v3_provenance(language, document)
        else:
            provenance_document = document["provenance"]
        return cls(
            schema_version=schema_version,
            model_version=model_version,
            language=language,
            supported_scripts=scripts,
            confidence_threshold=threshold,
            architecture=ByteModelConfig(**document["architecture"]),
            calibration=CalibrationConfig(
                slope=float(calibration_document["slope"]),
                intercept=float(calibration_document["intercept"]),
                population=ReferencePopulation(
                    provenance_document["calibration_population"]
                ),
                rows=int(calibration_document["rows"]),
            ),
            provenance=ModelProvenance(
                reference_population=ReferencePopulation(
                    provenance_document["reference_population"]
                ),
                label_source=LabelSource(provenance_document["label_source"]),
                training_seed=int(provenance_document["training_seed"]),
                normalization=str(provenance_document["normalization"]),
            ),
            reference_prior=_reference_prior(document),
        )


def _pinned_v3_provenance(language: str, document: dict[str, Any]) -> dict[str, Any]:
    """Normalize the immutable v3 release's original metadata schema."""
    if document.get("model_version") != "3.0" or "provenance" in document:
        raise ValueError("Unsupported schema 1 model metadata")
    training = document["training"]
    calibration = document["calibration"]
    if not isinstance(training, dict):
        raise ValueError("training metadata must be an object")
    if not isinstance(calibration, dict):
        raise ValueError("calibration metadata must be an object")
    if language == "eng":
        population = ReferencePopulation.SEPRI_HOUSEHOLD_HEADS
        label_source = LabelSource.LAND_CASTE_AND_SEPRI_RELIGION
        calibration_source = "SEPRI household heads, stable name-hash calibration split"
    else:
        population = ReferencePopulation.BIHAR_LAND_RECORD_NAMES
        label_source = LabelSource.LAND_CASTE
        calibration_source = (
            "Bihar land caste labels, stable name-hash calibration split"
        )
    if calibration.get("source") != calibration_source:
        raise ValueError("Unsupported schema 1 calibration metadata")
    return {
        "reference_population": population,
        "label_source": label_source,
        "calibration_population": population,
        "training_seed": training["seed"],
        "normalization": document["normalization"],
    }


def load_byte_classifier(path: Path, config: ByteModelConfig) -> ByteNameClassifier:
    """Load a v3 classifier from a non-executable safetensors artifact."""
    model = ByteNameClassifier(config)
    model.load_state_dict(load_file(path, device="cpu"), strict=True)
    model.eval()
    return model
