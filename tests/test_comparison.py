"""Regression tests for the v2-v3 aggregate comparison."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from pranaam.model_v3 import ByteModelConfig
from scripts.adhoc.compare_model_v2_v3 import (
    _prepare_comparison_splits,
    _survey_weight,
    _validate_split_counts,
    _weighted_metrics,
)


def test_comparison_reconstructs_splits_with_candidate_byte_limit() -> None:
    frame = pd.DataFrame(
        {
            "name": ["asha", "ali", "a" * 15],
            "label": [0.0, 1.0, 1.0],
        }
    )
    reconstructed = {
        name: frame.copy() for name in ("train", "validation", "calibration", "test")
    }
    document = {
        "training": {
            "splits": {
                "calibration": {"rows": 2, "muslim": 1},
                "test": {"rows": 2, "muslim": 1},
            }
        }
    }
    metadata = Mock(architecture=ByteModelConfig(max_bytes=16))

    with patch(
        "scripts.adhoc.compare_model_v2_v3.prepare_splits",
        return_value=reconstructed,
    ) as prepare_splits_mock:
        splits, tokenizer = _prepare_comparison_splits(
            document,
            metadata,
            Path("land"),
            Path("reds"),
            8.0,
        )

    assert tokenizer.max_content_bytes == 14
    assert splits["calibration"]["name"].tolist() == ["asha", "ali"]
    assert splits["test"]["name"].tolist() == ["asha", "ali"]
    prepare_splits_mock.assert_called_once_with(
        Path("land"),
        Path("reds"),
        "eng",
        survey_weight=8.0,
    )


def test_weighted_metrics_uses_artifact_confidence() -> None:
    probabilities = np.array([0.05, 0.15, 0.85, 0.95])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)

    at_eighty = _weighted_metrics(probabilities, labels, weights, 0.8)
    at_ninety = _weighted_metrics(probabilities, labels, weights, 0.9)

    assert at_eighty["coverage"] == 1.0
    assert at_ninety["coverage"] == 0.5


def test_survey_weight_reads_zero_mode_when_weight_is_absent() -> None:
    assert _survey_weight({"training": {"splits": {}}}) == 0.0
    assert _survey_weight({"training": {"survey_weight": 8.0}}) == 8.0


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_survey_weight_rejects_invalid_metadata(weight: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _survey_weight({"training": {"survey_weight": weight}})


def test_split_validation_rejects_mismatched_artifact_counts() -> None:
    frame = pd.DataFrame({"label": [0.0, 1.0]})
    splits = {"calibration": frame, "test": frame}
    document = {
        "training": {
            "splits": {
                "calibration": {"rows": 2, "muslim": 1},
                "test": {"rows": 3, "muslim": 1},
            }
        }
    }

    with pytest.raises(ValueError, match="test row count"):
        _validate_split_counts(document, splits)
