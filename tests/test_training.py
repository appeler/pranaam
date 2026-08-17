"""Tests for deterministic v3 training and evaluation helpers."""

import argparse

import numpy as np
import pandas as pd
import pytest
import torch

from pranaam.model_v3 import ByteModelConfig
from training.train_v3 import (
    _sigmoid,
    binary_log_loss,
    binary_metrics,
    confidence_threshold,
    fit_platt_scaler,
    limit_training_rows,
    stable_bucket,
    survey_weight,
    train_model,
)


def test_stable_bucket_is_deterministic_and_bounded() -> None:
    first = stable_bucket("shah rukh khan")

    assert first == stable_bucket("shah rukh khan")
    assert 0 <= first < 100
    assert stable_bucket("shah rukh khan", buckets=7) < 7


def test_platt_scaling_improves_miscalibrated_logits() -> None:
    logits = np.array([-4, -3, -2, -1, 1, 2, 3, 4], dtype=float) * 3
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

    slope, intercept = fit_platt_scaler(logits, labels)
    calibrated = _sigmoid(slope * logits + intercept)

    assert slope > 0
    assert binary_log_loss(calibrated, labels) <= binary_log_loss(
        _sigmoid(logits), labels
    )


def test_binary_metrics_reports_abstention_and_confusion_counts() -> None:
    probabilities = np.array([0.95, 0.75, 0.25, 0.05])
    labels = np.array([1, 1, 0, 0])

    metrics = binary_metrics(probabilities, labels, confidence_threshold=0.8)

    assert metrics["true_positive"] == 2
    assert metrics["true_negative"] == 2
    assert metrics["coverage"] == 0.5
    assert metrics["retained_accuracy"] == 1.0


@pytest.mark.parametrize("value", ["0.5", "0.4", "1.0"])
def test_confidence_threshold_rejects_unloadable_metadata(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=r"between 0\.5 and 1"):
        confidence_threshold(value)


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_survey_weight_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite and nonnegative"):
        survey_weight(value)


def test_training_row_limit_preserves_sources_and_labels() -> None:
    frame = pd.DataFrame(
        {
            "name": [f"name-{index}" for index in range(16)],
            "label": [0.0] * 4 + [1.0] * 4 + [0.0] * 4 + [1.0] * 4,
            "source": ["land"] * 8 + ["sepri"] * 8,
        }
    )

    first = limit_training_rows(frame, 8, seed=17)
    second = limit_training_rows(frame, 8, seed=17)

    assert first.equals(second)
    assert set(first[["source", "label"]].itertuples(index=False, name=None)) == {
        ("land", 0.0),
        ("land", 1.0),
        ("sepri", 0.0),
        ("sepri", 1.0),
    }


def test_training_row_limit_rejects_cap_smaller_than_strata() -> None:
    frame = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "label": [0.0, 1.0, 0.0, 1.0],
            "source": ["land", "land", "sepri", "sepri"],
        }
    )

    with pytest.raises(ValueError, match="at least 4"):
        limit_training_rows(frame, 3, seed=17)


def test_train_model_enables_deterministic_algorithms() -> None:
    train = pd.DataFrame(
        {
            "name": ["asha", "ali", "meera", "umar"],
            "label": [0.0, 1.0, 0.0, 1.0],
        }
    )
    config = ByteModelConfig(
        max_bytes=8,
        embedding_dim=2,
        hidden_dim=2,
        output_dim=2,
        dropout=0.0,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        train_model(
            train,
            train,
            config,
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            seed=17,
            device=torch.device("cpu"),
        )
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(previous)
