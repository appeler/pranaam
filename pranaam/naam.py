"""Public interface for calibrated name-pattern estimation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from .base import Base
from .logging import get_logger
from .model_v3 import (
    ByteNameClassifier,
    ByteTokenizer,
    ModelArtifactMetadata,
    load_byte_classifier,
    supports_name_script,
)
from .utils import MODEL_REVISION

logger = get_logger()

PREDICTION_BATCH_SIZE = 1024


def is_english(text: str) -> bool:
    """Return whether text contains only ASCII characters."""
    return text.isascii()


@dataclass(frozen=True)
class _LanguageModel:
    classifier: ByteNameClassifier
    tokenizer: ByteTokenizer
    metadata: ModelArtifactMetadata

    def predict(self, names: list[str]) -> np.ndarray:
        logits = []
        for start in range(0, len(names), PREDICTION_BATCH_SIZE):
            token_ids = self.tokenizer.encode(
                names[start : start + PREDICTION_BATCH_SIZE]
            )
            logits.append(self.classifier.predict_logits(token_ids).cpu().numpy())
        raw_logits = np.concatenate(logits, axis=0)
        calibration = self.metadata.calibration
        calibrated_logits = calibration.slope * raw_logits + calibration.intercept
        scores = np.empty_like(calibrated_logits, dtype=float)
        positive = calibrated_logits >= 0
        scores[positive] = 1 / (1 + np.exp(-calibrated_logits[positive]))
        exponent = np.exp(calibrated_logits[~positive])
        scores[~positive] = exponent / (1 + exponent)
        return scores


class Naam(Base):
    """Estimate binary name patterns for English or Hindi names."""

    _models: ClassVar[dict[str, _LanguageModel]] = {}
    _model_lock: ClassVar[RLock] = RLock()

    @classmethod
    def pred_rel(
        cls,
        names: str | list[str] | pd.Series,
        lang: Literal["eng", "hin"] = "eng",
        latest: bool = False,
    ) -> pd.DataFrame:
        """Return calibrated name-pattern estimates with abstention metadata.

        Args:
            names: One name, a list of names, or a pandas Series of names.
            lang: ``eng`` for the English model or ``hin`` for the Hindi model.
            latest: Refresh the pinned model artifacts in the local Hub cache.

        Returns:
            A data frame with calibrated scores, abstention and script status,
            and immutable model provenance.

        Raises:
            TypeError: If an input value is not a string.
            ValueError: If the language or input collection is invalid.
            RuntimeError: If model loading or inference fails.
        """
        if lang not in ("eng", "hin"):
            raise ValueError(f"Unsupported language: {lang}. Use 'eng' or 'hin'")

        if isinstance(names, str):
            name_list = [names]
        elif isinstance(names, pd.Series):
            name_list = names.tolist()
        else:
            name_list = list(names)

        if not name_list:
            raise ValueError("Input names list cannot be empty")
        for index, name in enumerate(name_list):
            if not isinstance(name, str):
                raise TypeError(f"Name at index {index} must be a string")
            if not name.strip():
                raise ValueError(
                    f"Name at index {index} is empty or contains only whitespace"
                )

        try:
            model = cls._model_for(lang, latest)
            supported = np.array(
                [
                    supports_name_script(name, model.metadata.supported_scripts)
                    for name in name_list
                ],
                dtype=bool,
            )
            scores = np.full(len(name_list), np.nan, dtype=float)
            if np.any(supported):
                supported_names = [
                    name
                    for name, is_supported in zip(name_list, supported, strict=True)
                    if is_supported
                ]
                supported_scores = np.asarray(
                    model.predict(supported_names), dtype=float
                )
                expected_shape = (len(supported_names),)
                if supported_scores.shape != expected_shape:
                    raise ValueError(
                        f"Model output must have shape {expected_shape}, got "
                        f"{supported_scores.shape}"
                    )
                if not np.all(np.isfinite(supported_scores)):
                    raise ValueError("Model output contains non-finite scores")
                if np.any((supported_scores < 0) | (supported_scores > 1)):
                    raise ValueError("Model scores must be between zero and one")
                scores[supported] = supported_scores

            threshold = model.metadata.confidence_threshold
            uncertain = supported & (scores > 1 - threshold) & (scores < threshold)
            abstained = ~supported | uncertain
            estimates = np.full(len(name_list), "uncertain", dtype=object)
            estimates[supported & (scores <= 1 - threshold)] = "not-muslim-associated"
            estimates[supported & (scores >= threshold)] = "muslim-associated"
            reasons: list[str | None] = []
            for is_supported, is_uncertain in zip(supported, uncertain, strict=True):
                if not is_supported:
                    reasons.append("unsupported-script")
                elif is_uncertain:
                    reasons.append("uncertain-score")
                else:
                    reasons.append(None)
            return pd.DataFrame(
                {
                    "name": name_list,
                    "name_pattern_estimate": estimates,
                    "muslim_score": pd.array(scores.tolist(), dtype="Float64"),
                    "abstained": abstained,
                    "abstention_reason": pd.array(reasons, dtype="string"),
                    "script_supported": supported,
                    "model_version": model.metadata.model_version,
                    "model_revision": MODEL_REVISION,
                }
            )
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise RuntimeError(f"Prediction failed: {exc}") from exc

    @classmethod
    def _model_for(
        cls, lang: Literal["eng", "hin"], latest: bool = False
    ) -> _LanguageModel:
        """Return the cached, language-specific model bundle."""
        with cls._model_lock:
            if latest or lang not in cls._models:
                cls._models[lang] = cls._load_model(lang, latest)
            return cls._models[lang]

    @classmethod
    def _load_model(
        cls, lang: Literal["eng", "hin"], latest: bool = False
    ) -> _LanguageModel:
        """Load and validate one language's tokenizer and classifier."""
        try:
            weights_path = cls.load_model_data(
                f"{lang}/model.safetensors", latest=latest
            )
            metadata_path = cls.load_model_data(f"{lang}/metadata.json", latest=latest)
            metadata = ModelArtifactMetadata.from_file(metadata_path)
            if metadata.language != lang:
                raise ValueError(
                    f"Model metadata language is {metadata.language!r}, "
                    f"expected {lang!r}"
                )
            logger.info("Loading %s model from %s", lang, weights_path)
            return _LanguageModel(
                classifier=load_byte_classifier(weights_path, metadata.architecture),
                tokenizer=ByteTokenizer(metadata.architecture.max_bytes),
                metadata=metadata,
            )
        except Exception as exc:
            logger.error("Failed to load %s model: %s", lang, exc)
            raise RuntimeError(f"Failed to load {lang} model: {exc}") from exc
