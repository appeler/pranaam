"""Main prediction interface for religion classification."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import ClassVar, Literal

import numpy as np
import pandas as pd

from .base import Base
from .logging import get_logger
from .model import ModelConfig, NameClassifier, NameTokenizer, load_classifier

logger = get_logger()

PREDICTION_BATCH_SIZE = 1024


def is_english(text: str) -> bool:
    """Return whether text contains only ASCII characters."""
    return text.isascii()


@dataclass(frozen=True)
class _LanguageModel:
    classifier: NameClassifier
    tokenizer: NameTokenizer

    def predict(self, names: list[str]) -> np.ndarray:
        probabilities = []
        for start in range(0, len(names), PREDICTION_BATCH_SIZE):
            token_ids = self.tokenizer.encode(
                names[start : start + PREDICTION_BATCH_SIZE]
            )
            probabilities.append(self.classifier.predict_proba(token_ids).cpu().numpy())
        return np.concatenate(probabilities, axis=0)


class Naam(Base):
    """Predict a binary religion label from English or Hindi names."""

    classes: ClassVar[tuple[str, str]] = ("not-muslim", "muslim")
    _config: ClassVar[ModelConfig] = ModelConfig()
    _models: ClassVar[dict[str, _LanguageModel]] = {}
    _model_lock: ClassVar[RLock] = RLock()

    @classmethod
    def pred_rel(
        cls,
        names: str | list[str] | pd.Series,
        lang: Literal["eng", "hin"] = "eng",
        latest: bool = False,
    ) -> pd.DataFrame:
        """Predict religion labels and Muslim-class probabilities.

        Args:
            names: One name, a list of names, or a pandas Series of names.
            lang: ``eng`` for the English model or ``hin`` for the Hindi model.
            latest: Refresh the pinned model artifacts in the local Hub cache.

        Returns:
            A data frame with the input, predicted label, and rounded Muslim
            probability percentage.

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
            probabilities = np.asarray(
                cls._model_for(lang, latest).predict(name_list), dtype=float
            )
            expected_shape = (len(name_list), len(cls.classes))
            if probabilities.shape != expected_shape:
                raise ValueError(
                    f"Model output must have shape {expected_shape}, got "
                    f"{probabilities.shape}"
                )
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("Model output contains non-finite probabilities")
            if np.any((probabilities < 0) | (probabilities > 1)):
                raise ValueError(
                    "Model output probabilities must be between zero and one"
                )
            if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError("Model output probabilities must sum to one")

            predictions = np.argmax(probabilities, axis=1)
            labels = [cls.classes[prediction] for prediction in predictions]
            muslim_probs = [
                float(np.around(probability[1] * 100)) for probability in probabilities
            ]
            return pd.DataFrame(
                {
                    "name": name_list,
                    "pred_label": labels,
                    "pred_prob_muslim": muslim_probs,
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
            vocabulary_path = cls.load_model_data(
                f"{lang}/vocabulary.txt", latest=latest
            )
            logger.info("Loading %s model from %s", lang, weights_path)
            return _LanguageModel(
                classifier=load_classifier(weights_path, cls._config),
                tokenizer=NameTokenizer.from_file(vocabulary_path, cls._config),
            )
        except Exception as exc:
            logger.error("Failed to load %s model: %s", lang, exc)
            raise RuntimeError(f"Failed to load {lang} model: {exc}") from exc
