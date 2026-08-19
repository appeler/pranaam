"""Public interface for calibrated name-pattern estimation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from torch import nn

from ._contract import ResultProvenance, metadata_columns, preserve_reserved_columns
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
SCORE_COLUMN = "muslim_score"

_TARGET = "muslim-name-pattern"
_INPUT_SCOPE = "full-name"


def is_english(text: str) -> bool:
    """Return whether text contains only ASCII characters."""
    return text.isascii()


def _calibrate(logits: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Apply the model's affine logit calibration and return probabilities."""
    calibrated = slope * logits + intercept
    scores = np.empty_like(calibrated, dtype=float)
    positive = calibrated >= 0
    scores[positive] = 1 / (1 + np.exp(-calibrated[positive]))
    exponent = np.exp(calibrated[~positive])
    scores[~positive] = exponent / (1 + exponent)
    return scores


def _shift_prior(
    scores: np.ndarray, reference_prior: float, target_prior: float
) -> np.ndarray:
    """Reweight calibrated probabilities to a different base rate.

    Multiplies the posterior odds by the ratio of target to reference prior
    odds, which is the correct adjustment when only the class balance, and
    not the class-conditional name distribution, differs between the two
    populations.
    """
    ratio = (target_prior / (1 - target_prior)) * (
        (1 - reference_prior) / reference_prior
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        odds = (scores / (1 - scores)) * ratio
        shifted = odds / (1 + odds)
    # A score of exactly one carries infinite odds, whose shifted limit is one.
    shifted = np.where(scores >= 1, 1.0, shifted)
    # A missing score stays missing. An abstaining row must never acquire one
    # here, and every arithmetic branch above turns NaN into something else.
    return np.where(np.isnan(scores), np.nan, shifted)


@dataclass(frozen=True)
class _LanguageModel:
    classifier: ByteNameClassifier
    tokenizer: ByteTokenizer
    metadata: ModelArtifactMetadata

    def _logits(self, names: list[str]) -> np.ndarray:
        logits = []
        for start in range(0, len(names), PREDICTION_BATCH_SIZE):
            token_ids = self.tokenizer.encode(
                names[start : start + PREDICTION_BATCH_SIZE]
            )
            logits.append(self.classifier.predict_logits(token_ids).cpu().numpy())
        return np.concatenate(logits, axis=0)

    def predict(self, names: list[str]) -> np.ndarray:
        """Return calibrated Muslim-pattern probabilities."""
        calibration = self.metadata.calibration
        return _calibrate(self._logits(names), calibration.slope, calibration.intercept)

    def predict_monte_carlo(self, names: list[str], iterations: int) -> np.ndarray:
        """Return calibrated probabilities from repeated dropout samples.

        Dropout stays active during the forward pass, so each iteration
        samples a different thinned network. The spread across iterations
        describes the model's own instability, not sampling error in the
        training data.
        """
        dropout_layers = [
            module
            for module in self.classifier.modules()
            if isinstance(module, nn.Dropout)
        ]
        was_training = [layer.training for layer in dropout_layers]
        for layer in dropout_layers:
            layer.train()
        try:
            calibration = self.metadata.calibration
            samples = np.stack(
                [
                    _calibrate(
                        self._logits(names), calibration.slope, calibration.intercept
                    )
                    for _ in range(iterations)
                ]
            )
        finally:
            for layer, training in zip(dropout_layers, was_training, strict=True):
                layer.train(training)
        return samples


@dataclass(frozen=True)
class _ClassifiedName:
    """One input name after normalization."""

    text: str
    script_supported: bool | None
    reason: str | None


class Naam(Base):
    """Estimate binary name patterns for English or Hindi names."""

    _models: ClassVar[dict[str, _LanguageModel]] = {}
    _model_lock: ClassVar[RLock] = RLock()

    @classmethod
    def estimate_muslim_name_pattern(
        cls,
        data: pd.DataFrame | pd.Series | list[str | None] | str,
        name_column: str | None = None,
        *,
        lang: Literal["eng", "hin"] = "eng",
        prior: float | None = None,
        uncertainty_level: float | None = None,
        mc_iterations: int = 64,
        refresh_pinned: bool = False,
    ) -> pd.DataFrame:
        """Estimate how far a name follows Muslim-associated naming patterns.

        The score is a calibrated probability on a 0 to 1 scale. Pranaam does
        not return a label: for a binary target the score carries the whole
        distribution, and the cutoff that would turn it into a decision
        depends on the caller's costs, not on this package.

        Args:
            data: DataFrame of inputs, or a name string, list, or Series.
            name_column: Column holding names for DataFrame input.
            lang: ``eng`` for the English model or ``hin`` for the Hindi model.
            prior: Share of the target population expected to carry
                Muslim-associated names. When given, scores are reweighted
                from the model's reference base rate to this one.
            uncertainty_level: Central interval to report from Monte Carlo
                dropout, such as ``0.9``. Omit for point estimates only.
            mc_iterations: Dropout samples drawn when ``uncertainty_level``
                is requested.
            refresh_pinned: Reload and verify the immutable model artifacts.
                Hub artifacts are redownloaded. Files under
                ``PRANAAM_MODEL_DIR`` are reread without network access.

        Returns:
            A copy of the input with the calibrated score, the contract's
            metadata columns, and, when requested, Monte Carlo summaries.

        Raises:
            ValueError: If an argument is outside its documented domain.
            RuntimeError: If model loading or inference fails.
        """
        if lang not in ("eng", "hin"):
            raise ValueError(f"Unsupported language: {lang}. Use 'eng' or 'hin'")
        if prior is not None and not 0 < prior < 1:
            raise ValueError("prior must be between zero and one, exclusive")
        if uncertainty_level is not None and not 0 < uncertainty_level < 1:
            raise ValueError(
                "uncertainty_level must be between zero and one, exclusive"
            )
        if uncertainty_level is not None and mc_iterations < 2:
            raise ValueError("mc_iterations must be at least two")

        frame, column = cls._prepare(data, name_column)
        try:
            model = cls._model_for(lang, refresh_pinned)
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise RuntimeError(f"Prediction failed: {exc}") from exc

        classified = [cls._classify(value, model) for value in frame[column].to_numpy()]
        scored = [item.reason is None for item in classified]
        rows = len(classified)
        scores = np.full(rows, np.nan)
        monte_carlo: np.ndarray | None = None
        usable = [row for row, ok in enumerate(scored) if ok]
        if usable:
            names = [classified[row].text for row in usable]
            try:
                point = np.asarray(model.predict(names), dtype=float)
                if point.shape != (len(names),):
                    raise ValueError(
                        f"Model output must have shape {(len(names),)}, "
                        f"got {point.shape}"
                    )
                if not np.all(np.isfinite(point)):
                    raise ValueError("Model output contains non-finite scores")
                if np.any((point < 0) | (point > 1)):
                    raise ValueError("Model scores must be between zero and one")
                if uncertainty_level is not None:
                    monte_carlo = np.full((mc_iterations, rows), np.nan)
                    monte_carlo[:, usable] = model.predict_monte_carlo(
                        names, mc_iterations
                    )
            except Exception as exc:
                logger.error("Prediction failed: %s", exc)
                raise RuntimeError(f"Prediction failed: {exc}") from exc
            scores[usable] = point

        reference_prior = model.metadata.reference_prior
        if prior is not None:
            if reference_prior is None:
                raise ValueError(
                    "this model artifact reports no reference base rate, so "
                    "prior shifting has nothing to shift from"
                )
            scores = _shift_prior(scores, reference_prior, prior)
            if monte_carlo is not None:
                monte_carlo = _shift_prior(monte_carlo, reference_prior, prior)

        values: dict[str, object] = {
            SCORE_COLUMN: pd.array(scores.tolist(), dtype="Float64"),
            "normalized_utf8_bytes": pd.array(
                [
                    model.tokenizer.normalized_utf8_length(item.text)
                    if item.text
                    else None
                    for item in classified
                ],
                dtype="Int64",
            ),
            "model_max_name_bytes": pd.array(
                [model.tokenizer.max_content_bytes] * rows, dtype="Int64"
            ),
            "model_language": pd.array(
                [model.metadata.language] * rows, dtype="string"
            ),
            "model_metadata_schema": pd.array(
                [model.metadata.schema_version] * rows, dtype="Int64"
            ),
            "label_source": pd.array(
                [model.metadata.provenance.label_source.value] * rows, dtype="string"
            ),
            "reference_prior": pd.array([reference_prior] * rows, dtype="Float64"),
            "target_prior": pd.array([prior] * rows, dtype="Float64"),
        }
        if monte_carlo is not None and uncertainty_level is not None:
            tail = (1 - uncertainty_level) / 2
            # Summarize only the scored columns. Abstaining rows hold an
            # all-missing sample, which every summary would reduce over an
            # empty slice.
            sampled = monte_carlo[:, usable]
            summaries = {
                f"{SCORE_COLUMN}_mc_mean": sampled.mean(axis=0),
                f"{SCORE_COLUMN}_mc_std": sampled.std(axis=0, ddof=1),
                f"{SCORE_COLUMN}_mc_lower": np.quantile(sampled, tail, axis=0),
                f"{SCORE_COLUMN}_mc_upper": np.quantile(sampled, 1 - tail, axis=0),
            }
            for name, summary in summaries.items():
                column = np.full(rows, np.nan)
                column[usable] = summary
                values[name] = pd.array(column.tolist(), dtype="Float64")

        provenance = ResultProvenance(
            target=_TARGET,
            input_scope=_INPUT_SCOPE,
            model_id=f"pranaam-byte-classifier-{lang}",
            model_version=str(model.metadata.model_version),
            model_revision=MODEL_REVISION,
            reference_population=(model.metadata.provenance.reference_population.value),
            calibration_status="platt-scaled",
            calibration_reference=model.metadata.calibration.population.value,
            uncertainty_method=(
                "monte-carlo-dropout" if uncertainty_level is not None else None
            ),
            uncertainty_level=uncertainty_level,
        )
        metadata = metadata_columns(
            provenance,
            scored=scored,
            script_supported=[item.script_supported for item in classified],
            abstention_reasons=[item.reason for item in classified],
        )
        result = preserve_reserved_columns(frame, [*values, *metadata])
        for name, column_values in {**values, **metadata}.items():
            result[name] = column_values
        return result

    @classmethod
    def _prepare(
        cls,
        data: pd.DataFrame | pd.Series | list[str | None] | str,
        name_column: str | None,
    ) -> tuple[pd.DataFrame, str]:
        """Validate input and return the working frame and its name column."""
        if isinstance(data, pd.DataFrame):
            if name_column is None:
                raise ValueError("name_column is required for DataFrame input")
            if list(data.columns).count(name_column) != 1:
                raise ValueError(
                    f"name column {name_column!r} must appear exactly once"
                )
            return data, name_column
        if isinstance(data, str):
            return pd.DataFrame({"name": [data]}), "name"
        if isinstance(data, pd.Series):
            return data.to_frame(name="name"), "name"
        if isinstance(data, list):
            return pd.DataFrame({"name": data}), "name"
        raise TypeError("data must be a DataFrame, Series, list, or string")

    @classmethod
    def _classify(cls, value: object, model: _LanguageModel) -> _ClassifiedName:
        """Map one raw input to a usable name or an abstention reason."""
        if not isinstance(value, str) or not value.strip():
            return _ClassifiedName("", None, "missing-name")
        text = value.strip()
        if not any(character.isalpha() for character in text):
            return _ClassifiedName("", None, "no-letters")
        if not supports_name_script(text, model.metadata.supported_scripts):
            return _ClassifiedName(text, False, "unsupported-script")
        if model.tokenizer.truncates(text):
            return _ClassifiedName(text, True, "input-truncated")
        return _ClassifiedName(text, True, None)

    @classmethod
    def _model_for(
        cls, lang: Literal["eng", "hin"], refresh_pinned: bool = False
    ) -> _LanguageModel:
        """Return the cached, language-specific model bundle."""
        with cls._model_lock:
            if refresh_pinned or lang not in cls._models:
                cls._models[lang] = cls._load_model(lang, refresh_pinned)
            return cls._models[lang]

    @classmethod
    def _load_model(
        cls, lang: Literal["eng", "hin"], refresh_pinned: bool = False
    ) -> _LanguageModel:
        """Load and validate one language's tokenizer and classifier."""
        try:
            weights_path = cls.load_model_data(
                f"{lang}/model.safetensors", refresh_pinned=refresh_pinned
            )
            metadata_path = cls.load_model_data(
                f"{lang}/metadata.json", refresh_pinned=refresh_pinned
            )
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
