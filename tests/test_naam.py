"""Tests for the public prediction interface and model cache."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import torch

from pranaam.model_v3 import (
    ByteModelConfig,
    CalibrationConfig,
    ModelArtifactMetadata,
)
from pranaam.naam import Naam, _LanguageModel, is_english
from pranaam.utils import MODEL_REVISION


def metadata(
    language: str = "eng", scripts: tuple[str, ...] = ("LATIN",)
) -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        model_version="3.0",
        language=language,
        supported_scripts=scripts,
        confidence_threshold=0.8,
        architecture=ByteModelConfig(),
        calibration=CalibrationConfig(
            slope=1,
            intercept=0,
            source="test",
            rows=10,
        ),
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Shah Rukh Khan", True),
        ("123 ABC!", True),
        ("", True),
        ("शाहरुख खान", False),
        ("Khan खान", False),
    ],
)
def test_is_english(text: str, expected: bool) -> None:
    """ASCII-only detection handles English, Hindi, mixed, and empty text."""
    assert is_english(text) is expected


@patch("pranaam.naam.PREDICTION_BATCH_SIZE", 2)
def test_language_model_batches_predictions() -> None:
    """Inference keeps model inputs bounded and preserves their order."""
    tokenizer = Mock()
    tokenizer.encode.side_effect = [
        torch.tensor([[0], [1]]),
        torch.tensor([[2], [3]]),
        torch.tensor([[4]]),
    ]
    classifier = Mock()
    classifier.predict_logits.side_effect = [
        torch.tensor([-2.0, -1.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([2.0]),
    ]
    model = _LanguageModel(
        classifier=classifier,
        tokenizer=tokenizer,
        metadata=metadata(),
    )

    probabilities = model.predict(["one", "two", "three", "four", "five"])

    assert tokenizer.encode.call_args_list == [
        call(["one", "two"]),
        call(["three", "four"]),
        call(["five"]),
    ]
    np.testing.assert_allclose(
        probabilities,
        1 / (1 + np.exp(-np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))),
    )


@patch.object(Naam, "_model_for")
def test_single_string_input(mock_model_for: Mock) -> None:
    """A single name becomes a one-row result."""
    model = Mock()
    model.metadata = metadata()
    model.predict.return_value = np.array([0.8])
    mock_model_for.return_value = model

    result = Naam.pred_rel("Test Name")

    assert result.to_dict(orient="records") == [
        {
            "name": "Test Name",
            "name_pattern_estimate": "muslim-associated",
            "muslim_score": 0.8,
            "abstained": False,
            "abstention_reason": None,
            "script_supported": True,
            "model_version": "3.0",
            "model_revision": MODEL_REVISION,
        }
    ]
    model.predict.assert_called_once_with(["Test Name"])


@patch.object(Naam, "_model_for")
def test_list_and_series_inputs(mock_model_for: Mock) -> None:
    """Lists and pandas Series retain their row order."""
    model = Mock()
    model.metadata = metadata()
    model.predict.return_value = np.array([0.9, 0.1])
    mock_model_for.return_value = model

    names = ["Name One", "Name Two"]
    list_result = Naam.pred_rel(names)
    series_result = Naam.pred_rel(pd.Series(names))

    assert list(list_result["name"]) == names
    assert list(series_result["name"]) == names
    assert list_result["name_pattern_estimate"].tolist() == [
        "muslim-associated",
        "not-muslim-associated",
    ]


@patch.object(Naam, "_model_for")
def test_uncertain_and_unsupported_names_abstain(mock_model_for: Mock) -> None:
    model = Mock()
    model.metadata = metadata()
    model.predict.return_value = np.array([0.5])
    mock_model_for.return_value = model

    result = Naam.pred_rel(["Shared Name", "محمد خان"])

    assert result["name_pattern_estimate"].tolist() == ["uncertain", "uncertain"]
    assert result["abstained"].tolist() == [True, True]
    assert result["abstention_reason"].tolist() == [
        "uncertain-score",
        "unsupported-script",
    ]
    assert result["script_supported"].tolist() == [True, False]
    assert result.iloc[0]["muslim_score"] == 0.5
    assert pd.isna(result.iloc[1]["muslim_score"])
    model.predict.assert_called_once_with(["Shared Name"])


@pytest.mark.parametrize("names", [[], "", "   "])
def test_empty_inputs_are_rejected(names: object) -> None:
    """Empty collections, strings, and whitespace fail before model loading."""
    with pytest.raises(ValueError, match="empty"):
        Naam.pred_rel(names)  # type: ignore[arg-type]


@pytest.mark.parametrize("names", [["Valid", None], pd.Series(["Valid", 1])])
def test_nonstring_values_are_rejected(names: object) -> None:
    """Mixed collections report the offending input position."""
    with pytest.raises(TypeError, match="index 1 must be a string"):
        Naam.pred_rel(names)  # type: ignore[arg-type]


def test_invalid_language_is_rejected() -> None:
    """Only published language models can be requested."""
    with pytest.raises(ValueError, match="Unsupported language"):
        Naam.pred_rel("Test", lang="fra")  # type: ignore[arg-type]


@patch.object(Naam, "_model_for")
@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.array([0.5, 0.8]), "must have shape"),
        (np.array([np.nan]), "non-finite"),
        (np.array([-0.1]), "between zero and one"),
        (np.array([1.1]), "between zero and one"),
    ],
)
def test_invalid_model_probabilities_fail(
    mock_model_for: Mock, scores: np.ndarray, message: str
) -> None:
    """Malformed model output cannot be presented as a prediction."""
    model = Mock()
    model.metadata = metadata()
    model.predict.return_value = scores
    mock_model_for.return_value = model

    with pytest.raises(RuntimeError, match=message):
        Naam.pred_rel(["Test"])


@patch.object(Naam, "load_model_data")
@patch("pranaam.naam.ModelArtifactMetadata.from_file")
@patch("pranaam.naam.load_byte_classifier")
def test_load_model_assembles_verified_bundle(
    mock_load_classifier: Mock,
    mock_metadata_from_file: Mock,
    mock_load_data: Mock,
) -> None:
    """Weights and inference metadata are loaded from pinned artifacts."""
    weights = Path("/cache/eng/model.safetensors")
    metadata_path = Path("/cache/eng/metadata.json")
    classifier = Mock()
    model_metadata = metadata()
    mock_load_data.side_effect = [weights, metadata_path]
    mock_load_classifier.return_value = classifier
    mock_metadata_from_file.return_value = model_metadata

    bundle = Naam._load_model("eng", latest=True)

    assert mock_load_data.call_args_list == [
        call("eng/model.safetensors", latest=True),
        call("eng/metadata.json", latest=True),
    ]
    mock_metadata_from_file.assert_called_once_with(metadata_path)
    mock_load_classifier.assert_called_once_with(weights, model_metadata.architecture)
    assert bundle.classifier is classifier
    assert bundle.metadata is model_metadata
    assert bundle.tokenizer.max_bytes == model_metadata.architecture.max_bytes


@patch.object(Naam, "load_model_data", side_effect=OSError("offline"))
def test_load_model_wraps_artifact_failure(mock_load_data: Mock) -> None:
    """Loading failures identify the requested language."""
    with pytest.raises(RuntimeError, match="Failed to load hin model"):
        Naam._load_model("hin")
    mock_load_data.assert_called_once()


@patch.object(Naam, "_load_model")
def test_models_are_cached_by_language(mock_load_model: Mock) -> None:
    """Each language loads once and remains independently cached."""
    english_model = Mock()
    hindi_model = Mock()
    english_model.metadata = metadata()
    hindi_model.metadata = metadata("hin", ("DEVANAGARI",))
    mock_load_model.side_effect = [english_model, hindi_model]

    assert Naam._model_for("eng") is english_model
    assert Naam._model_for("eng") is english_model
    assert Naam._model_for("hin") is hindi_model
    assert Naam._model_for("hin") is hindi_model
    assert mock_load_model.call_args_list == [call("eng", False), call("hin", False)]


@patch.object(Naam, "_load_model")
def test_latest_replaces_only_requested_language(mock_load_model: Mock) -> None:
    """Refreshing English does not evict an already loaded Hindi model."""
    old_english = Mock()
    hindi = Mock()
    new_english = Mock()
    Naam._models.update(eng=old_english, hin=hindi)
    mock_load_model.return_value = new_english

    assert Naam._model_for("eng", latest=True) is new_english
    assert Naam._model_for("hin") is hindi
    mock_load_model.assert_called_once_with("eng", True)


@patch.object(Naam, "_load_model")
def test_concurrent_requests_load_one_shared_model(mock_load_model: Mock) -> None:
    """Concurrent first requests for one language perform one model load."""
    loading = Event()
    release_load = Event()
    english_model = Mock()

    def load_model(lang: str, latest: bool) -> Mock:
        loading.set()
        assert release_load.wait(timeout=2)
        return english_model

    mock_load_model.side_effect = load_model
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(Naam._model_for, "eng")
        assert loading.wait(timeout=2)
        second = executor.submit(Naam._model_for, "eng")
        release_load.set()
        assert first.result(timeout=2) is english_model
        assert second.result(timeout=2) is english_model

    mock_load_model.assert_called_once_with("eng", False)


@patch.object(Naam, "_load_model")
def test_concurrent_languages_keep_their_predictions(
    mock_load_model: Mock,
) -> None:
    """A Hindi load cannot redirect an in-flight English prediction."""
    english_predicting = Event()
    hindi_loaded = Event()
    english_model = Mock()
    hindi_model = Mock()

    def predict_english(names: list[str]) -> np.ndarray:
        english_predicting.set()
        assert hindi_loaded.wait(timeout=2)
        return np.array([0.95])

    english_model.predict.side_effect = predict_english
    english_model.metadata = metadata()
    hindi_model.predict.return_value = np.array([0.1])
    hindi_model.metadata = metadata("hin", ("DEVANAGARI",))

    def load_model(lang: str, latest: bool) -> Mock:
        if lang == "hin":
            hindi_loaded.set()
            return hindi_model
        return english_model

    mock_load_model.side_effect = load_model
    with ThreadPoolExecutor(max_workers=2) as executor:
        english_future = executor.submit(Naam.pred_rel, ["Shah"], "eng")
        assert english_predicting.wait(timeout=2)
        hindi_future = executor.submit(Naam.pred_rel, ["अमिताभ"], "hin")
        english_result = english_future.result(timeout=2)
        hindi_result = hindi_future.result(timeout=2)

    assert english_result.iloc[0]["name_pattern_estimate"] == "muslim-associated"
    assert hindi_result.iloc[0]["name_pattern_estimate"] == "not-muslim-associated"


@patch.object(Naam, "_model_for")
def test_prediction_error_preserves_cause(mock_model_for: Mock) -> None:
    """Inference failures surface through the documented runtime error."""
    model = Mock()
    model.metadata = metadata()
    model.predict.side_effect = ValueError("bad tensor")
    mock_model_for.return_value = model

    with pytest.raises(RuntimeError, match="Prediction failed: bad tensor"):
        Naam.pred_rel(["Test"])
