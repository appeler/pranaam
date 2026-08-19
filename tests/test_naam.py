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
    ByteTokenizer,
    CalibrationConfig,
    LabelSource,
    ModelArtifactMetadata,
    ModelProvenance,
    ReferencePopulation,
)
from pranaam.naam import Naam, _LanguageModel, is_english
from pranaam.utils import MODEL_REVISION


def metadata(
    language: str = "eng", scripts: tuple[str, ...] = ("LATIN",)
) -> ModelArtifactMetadata:
    return ModelArtifactMetadata(
        schema_version=2,
        model_version="3.0",
        language=language,
        supported_scripts=scripts,
        confidence_threshold=0.8,
        architecture=ByteModelConfig(),
        calibration=CalibrationConfig(
            slope=1,
            intercept=0,
            population=(
                ReferencePopulation.SEPRI_HOUSEHOLD_HEADS
                if language == "eng"
                else ReferencePopulation.BIHAR_LAND_RECORD_NAMES
            ),
            rows=10,
        ),
        provenance=ModelProvenance(
            reference_population=(
                ReferencePopulation.SEPRI_HOUSEHOLD_HEADS
                if language == "eng"
                else ReferencePopulation.BIHAR_LAND_RECORD_NAMES
            ),
            label_source=(
                LabelSource.LAND_CASTE_AND_SEPRI_RELIGION
                if language == "eng"
                else LabelSource.LAND_CASTE
            ),
            training_seed=7,
            normalization="Unicode NFKC, casefold, collapse whitespace",
        ),
        reference_prior=0.1,
    )


def mock_language_model(
    model_metadata: ModelArtifactMetadata | None = None,
) -> Mock:
    """Return a model mock with a real serving tokenizer contract."""
    selected_metadata = model_metadata or metadata()
    model = Mock()
    model.metadata = selected_metadata
    model.tokenizer = ByteTokenizer(selected_metadata.architecture.max_bytes)
    return model


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
    model = mock_language_model()
    model.predict.return_value = np.array([0.8])
    mock_model_for.return_value = model

    result = Naam.estimate_muslim_name_pattern("Test Name")

    assert len(result) == 1
    row = result.iloc[0]
    assert row["name"] == "Test Name"
    assert row["muslim_score"] == 0.8
    assert bool(row["scored"])
    assert not bool(row["abstained"])
    assert pd.isna(row["abstention_reason"])
    assert bool(row["script_supported"])
    assert row["normalized_utf8_bytes"] == 9
    assert row["model_max_name_bytes"] == 126
    assert row["reference_population"] == "SEPRI household heads"
    assert row["label_source"] == (
        "Bihar land caste/community labels and SEPRI household religion"
    )
    assert row["calibration_reference"] == "SEPRI household heads"
    assert row["model_language"] == "eng"
    assert row["model_version"] == "3.0"
    assert row["model_revision"] == MODEL_REVISION
    assert row["inference_contract_version"] == "1.1"
    assert row["result_form"] == "score"
    assert row["target"] == "muslim-name-pattern"
    assert row["calibration_status"] == "platt-scaled"
    assert pd.isna(row["uncertainty_method"])
    assert "name_pattern_estimate" not in result.columns
    assert "predicted_label" not in result.columns
    model.predict.assert_called_once_with(["Test Name"])


@patch.object(Naam, "_model_for")
def test_list_and_series_inputs(mock_model_for: Mock) -> None:
    """Lists and pandas Series retain their row order."""
    model = mock_language_model()
    model.predict.return_value = np.array([0.9, 0.1])
    mock_model_for.return_value = model

    names = ["Name One", "Name Two"]
    list_result = Naam.estimate_muslim_name_pattern(names)
    series_result = Naam.estimate_muslim_name_pattern(pd.Series(names))

    assert list(list_result["name"]) == names
    assert list(series_result["name"]) == names
    assert list_result["muslim_score"].tolist() == [0.9, 0.1]
    assert series_result["muslim_score"].tolist() == [0.9, 0.1]


@patch.object(Naam, "_model_for")
def test_midrange_scores_are_returned_and_unsupported_scripts_abstain(
    mock_model_for: Mock,
) -> None:
    """Score form withholds nothing it can compute, however mid-range."""
    model = mock_language_model()
    model.predict.return_value = np.array([0.5])
    mock_model_for.return_value = model

    result = Naam.estimate_muslim_name_pattern(["Shared Name", "محمد خان"])

    assert result["abstained"].tolist() == [False, True]
    assert pd.isna(result.iloc[0]["abstention_reason"])
    assert result.iloc[1]["abstention_reason"] == "unsupported-script"
    assert result["script_supported"].tolist() == [True, False]
    assert result.iloc[0]["muslim_score"] == 0.5
    assert pd.isna(result.iloc[1]["muslim_score"])
    model.predict.assert_called_once_with(["Shared Name"])


@patch.object(Naam, "_model_for")
def test_utf8_byte_truncation_is_an_explicit_abstention(mock_model_for: Mock) -> None:
    model = mock_language_model()
    mock_model_for.return_value = model
    name = "é" * 64

    result = Naam.estimate_muslim_name_pattern(name)

    assert result.iloc[0]["normalized_utf8_bytes"] == 128
    assert result.iloc[0]["abstained"]
    assert result.iloc[0]["abstention_reason"] == "input-truncated"
    assert pd.isna(result.iloc[0]["muslim_score"])
    model.predict.assert_not_called()


@patch.object(Naam, "_model_for")
def test_empty_collection_returns_an_empty_frame(mock_model_for: Mock) -> None:
    """No rows in means no rows out, not an exception."""
    mock_model_for.return_value = mock_language_model()

    result = Naam.estimate_muslim_name_pattern([])

    assert len(result) == 0
    assert "muslim_score" in result.columns


@pytest.mark.parametrize("value", ["", "   "])
@patch.object(Naam, "_model_for")
def test_blank_names_abstain(mock_model_for: Mock, value: str) -> None:
    """A blank cell is data the package cannot score, not a caller error."""
    mock_model_for.return_value = mock_language_model()

    result = Naam.estimate_muslim_name_pattern([value])

    assert not bool(result.iloc[0]["scored"])
    assert result.iloc[0]["abstention_reason"] == "missing-name"


@pytest.mark.parametrize("names", [["Valid", None], ["Valid", 1]])
@patch.object(Naam, "_model_for")
def test_nonstring_values_abstain(mock_model_for: Mock, names: list[object]) -> None:
    """Mixed collections abstain per row instead of failing the whole call."""
    model = mock_language_model()
    model.predict.return_value = np.array([0.7])
    mock_model_for.return_value = model

    result = Naam.estimate_muslim_name_pattern(names)  # type: ignore[arg-type]

    assert result["scored"].tolist() == [True, False]
    assert result.iloc[1]["abstention_reason"] == "missing-name"
    model.predict.assert_called_once_with(["Valid"])


def test_invalid_language_is_rejected() -> None:
    """Only published language models can be requested."""
    with pytest.raises(ValueError, match="Unsupported language"):
        Naam.estimate_muslim_name_pattern("Test", lang="fra")  # type: ignore[arg-type]


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
    model = mock_language_model()
    model.predict.return_value = scores
    mock_model_for.return_value = model

    with pytest.raises(RuntimeError, match=message):
        Naam.estimate_muslim_name_pattern(["Test"])


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

    bundle = Naam._load_model("eng", refresh_pinned=True)

    assert mock_load_data.call_args_list == [
        call("eng/model.safetensors", refresh_pinned=True),
        call("eng/metadata.json", refresh_pinned=True),
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
    english_model = mock_language_model()
    hindi_model = mock_language_model(metadata("hin", ("DEVANAGARI",)))
    mock_load_model.side_effect = [english_model, hindi_model]

    assert Naam._model_for("eng") is english_model
    assert Naam._model_for("eng") is english_model
    assert Naam._model_for("hin") is hindi_model
    assert Naam._model_for("hin") is hindi_model
    assert mock_load_model.call_args_list == [call("eng", False), call("hin", False)]


@patch.object(Naam, "_load_model")
def test_refresh_pinned_replaces_only_requested_language(mock_load_model: Mock) -> None:
    """Refreshing English does not evict an already loaded Hindi model."""
    old_english = Mock()
    hindi = Mock()
    new_english = Mock()
    Naam._models.update(eng=old_english, hin=hindi)
    mock_load_model.return_value = new_english

    assert Naam._model_for("eng", refresh_pinned=True) is new_english
    assert Naam._model_for("hin") is hindi
    mock_load_model.assert_called_once_with("eng", True)


@patch.object(Naam, "_load_model")
def test_concurrent_requests_load_one_shared_model(mock_load_model: Mock) -> None:
    """Concurrent first requests for one language perform one model load."""
    loading = Event()
    release_load = Event()
    english_model = Mock()

    def load_model(lang: str, refresh_pinned: bool) -> Mock:
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
    english_model = mock_language_model()
    hindi_model = mock_language_model(metadata("hin", ("DEVANAGARI",)))

    def predict_english(names: list[str]) -> np.ndarray:
        english_predicting.set()
        assert hindi_loaded.wait(timeout=2)
        return np.array([0.95])

    english_model.predict.side_effect = predict_english
    hindi_model.predict.return_value = np.array([0.1])

    def load_model(lang: str, refresh_pinned: bool) -> Mock:
        if lang == "hin":
            hindi_loaded.set()
            return hindi_model
        return english_model

    mock_load_model.side_effect = load_model
    with ThreadPoolExecutor(max_workers=2) as executor:
        english_future = executor.submit(
            Naam.estimate_muslim_name_pattern, ["Shah"], lang="eng"
        )
        assert english_predicting.wait(timeout=2)
        hindi_future = executor.submit(
            Naam.estimate_muslim_name_pattern, ["अमिताभ"], lang="hin"
        )
        english_result = english_future.result(timeout=2)
        hindi_result = hindi_future.result(timeout=2)

    assert english_result.iloc[0]["muslim_score"] == pytest.approx(0.95)
    assert hindi_result.iloc[0]["muslim_score"] == pytest.approx(0.1)


@patch.object(Naam, "_model_for")
def test_prediction_error_preserves_cause(mock_model_for: Mock) -> None:
    """Inference failures surface through the documented runtime error."""
    model = mock_language_model()
    model.predict.side_effect = ValueError("bad tensor")
    mock_model_for.return_value = model

    with pytest.raises(RuntimeError, match="Prediction failed: bad tensor"):
        Naam.estimate_muslim_name_pattern(["Test"])
