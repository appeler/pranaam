"""Tests for the public prediction interface and model cache."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import torch

from pranaam.naam import Naam, _LanguageModel, is_english


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
    classifier.predict_proba.side_effect = [
        torch.tensor([[0.9, 0.1], [0.8, 0.2]]),
        torch.tensor([[0.7, 0.3], [0.6, 0.4]]),
        torch.tensor([[0.5, 0.5]]),
    ]
    model = _LanguageModel(classifier=classifier, tokenizer=tokenizer)

    probabilities = model.predict(["one", "two", "three", "four", "five"])

    assert tokenizer.encode.call_args_list == [
        call(["one", "two"]),
        call(["three", "four"]),
        call(["five"]),
    ]
    np.testing.assert_allclose(
        probabilities,
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
        ],
    )


@patch.object(Naam, "_model_for")
def test_single_string_input(mock_model_for: Mock) -> None:
    """A single name becomes a one-row result."""
    model = Mock()
    model.predict.return_value = np.array([[0.2, 0.8]])
    mock_model_for.return_value = model

    result = Naam.pred_rel("Test Name")

    assert result.to_dict(orient="records") == [
        {
            "name": "Test Name",
            "pred_label": "muslim",
            "pred_prob_muslim": 80.0,
        }
    ]
    model.predict.assert_called_once_with(["Test Name"])


@patch.object(Naam, "_model_for")
def test_list_and_series_inputs(mock_model_for: Mock) -> None:
    """Lists and pandas Series retain their row order."""
    model = Mock()
    model.predict.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
    mock_model_for.return_value = model

    names = ["Name One", "Name Two"]
    list_result = Naam.pred_rel(names)
    series_result = Naam.pred_rel(pd.Series(names))

    assert list(list_result["name"]) == names
    assert list(series_result["name"]) == names
    assert list_result["pred_label"].tolist() == ["muslim", "not-muslim"]


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
    ("probabilities", "message"),
    [
        (np.array([[0.5, 0.5], [0.2, 0.8]]), "must have shape"),
        (np.array([[np.nan, np.nan]]), "non-finite"),
        (np.array([[-0.1, 1.1]]), "between zero and one"),
        (np.array([[0.2, 0.2]]), "sum to one"),
    ],
)
def test_invalid_model_probabilities_fail(
    mock_model_for: Mock, probabilities: np.ndarray, message: str
) -> None:
    """Malformed model output cannot be presented as a prediction."""
    model = Mock()
    model.predict.return_value = probabilities
    mock_model_for.return_value = model

    with pytest.raises(RuntimeError, match=message):
        Naam.pred_rel(["Test"])


@patch.object(Naam, "load_model_data")
@patch("pranaam.naam.NameTokenizer.from_file")
@patch("pranaam.naam.load_classifier")
def test_load_model_assembles_verified_bundle(
    mock_load_classifier: Mock,
    mock_tokenizer_from_file: Mock,
    mock_load_data: Mock,
) -> None:
    """Weights and vocabulary are loaded from separate pinned artifacts."""
    weights = Path("/cache/eng/model.safetensors")
    vocabulary = Path("/cache/eng/vocabulary.txt")
    classifier = Mock()
    tokenizer = Mock()
    mock_load_data.side_effect = [weights, vocabulary]
    mock_load_classifier.return_value = classifier
    mock_tokenizer_from_file.return_value = tokenizer

    bundle = Naam._load_model("eng", latest=True)

    assert mock_load_data.call_args_list == [
        call("eng/model.safetensors", latest=True),
        call("eng/vocabulary.txt", latest=True),
    ]
    mock_load_classifier.assert_called_once_with(weights, Naam._config)
    mock_tokenizer_from_file.assert_called_once_with(vocabulary, Naam._config)
    assert bundle.classifier is classifier
    assert bundle.tokenizer is tokenizer


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
        return np.array([[0.05, 0.95]])

    english_model.predict.side_effect = predict_english
    hindi_model.predict.return_value = np.array([[0.9, 0.1]])

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

    assert english_result.iloc[0]["pred_label"] == "muslim"
    assert hindi_result.iloc[0]["pred_label"] == "not-muslim"


@patch.object(Naam, "_model_for")
def test_prediction_error_preserves_cause(mock_model_for: Mock) -> None:
    """Inference failures surface through the documented runtime error."""
    model = Mock()
    model.predict.side_effect = ValueError("bad tensor")
    mock_model_for.return_value = model

    with pytest.raises(RuntimeError, match="Prediction failed: bad tensor"):
        Naam.pred_rel(["Test"])
