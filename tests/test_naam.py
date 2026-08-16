"""Comprehensive tests for naam module."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from pranaam.naam import Naam, is_english


class TestIsEnglish:
    """Test the is_english utility function."""

    def test_english_text(self) -> None:
        """Test detection of English text."""
        assert is_english("Hello World") is True
        assert is_english("Shah Rukh Khan") is True
        assert is_english("123 ABC") is True

    def test_hindi_text(self) -> None:
        """Test detection of Hindi text."""
        assert is_english("शाहरुख खान") is False
        assert is_english("अमिताभ बच्चन") is False
        assert is_english("हैलो वर्ल्ड") is False

    def test_mixed_text(self) -> None:
        """Test mixed text (contains non-ASCII)."""
        assert is_english("Hello शाहरुख") is False
        assert is_english("Khan खान") is False

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert is_english("") is True  # Empty string is ASCII

    def test_special_characters(self) -> None:
        """Test special characters."""
        assert is_english("Hello! @#$%") is True
        assert is_english("Test\nLine") is True


class TestNaamValidation:
    """Test input validation for Naam class."""

    # Removed validation tests that call Naam.pred_rel directly - could trigger model loading


class TestNaamInputHandling:
    """Test input handling and conversion in Naam class."""

    @patch.object(Naam, "_model_for")
    def test_single_string_input(self, mock_model_for: Mock) -> None:
        """Test handling of single string input."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.2, 0.8]])
        mock_model_for.return_value = mock_model

        result = Naam.pred_rel("Test Name")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Test Name"
        mock_model.predict.assert_called_once_with(["Test Name"], verbose=0)

    @patch.object(Naam, "_model_for")
    def test_list_input(self, mock_model_for: Mock) -> None:
        """Test handling of list input."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        mock_model_for.return_value = mock_model

        names = ["Name One", "Name Two"]
        result = Naam.pred_rel(names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["name"]) == names

    @patch.object(Naam, "_model_for")
    def test_pandas_series_input(self, mock_model_for: Mock) -> None:
        """Test handling of pandas Series input."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.2, 0.8]])
        mock_model_for.return_value = mock_model

        names = pd.Series(["Test Name"])
        result = Naam.pred_rel(names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestNaamPredictions:
    """Test prediction functionality."""

    @patch.object(Naam, "_model_for")
    def test_prediction_output_structure(self, mock_model_for: Mock) -> None:
        """Test that predictions return correct DataFrame structure."""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        mock_model_for.return_value = mock_model

        names = ["Name One", "Name Two"]
        result = Naam.pred_rel(names)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["name", "pred_label", "pred_prob_muslim"]
        assert len(result) == 2

        # Check data types
        assert pd.api.types.is_string_dtype(result["name"].dtype)
        assert pd.api.types.is_string_dtype(result["pred_label"].dtype)
        assert pd.api.types.is_numeric_dtype(result["pred_prob_muslim"])

    @patch.object(Naam, "_model_for")
    def test_prediction_labels(self, mock_model_for: Mock) -> None:
        """Test that predictions use correct labels."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        mock_model_for.return_value = mock_model

        result = Naam.pred_rel(["Name One", "Name Two"])

        # First prediction: higher muslim probability -> "muslim"
        # Second prediction: higher not-muslim probability -> "not-muslim"
        assert result.iloc[0]["pred_label"] == "muslim"
        assert result.iloc[1]["pred_label"] == "not-muslim"

    @patch.object(Naam, "_model_for")
    def test_prediction_probabilities(self, mock_model_for: Mock) -> None:
        """Test that model probabilities are not transformed a second time."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.0465, 0.9535]])
        mock_model_for.return_value = mock_model

        result = Naam.pred_rel(["Test Name"])

        assert result.iloc[0]["pred_prob_muslim"] == 95.0

    @patch.object(Naam, "_model_for")
    def test_invalid_model_probabilities_fail(self, mock_model_for: Mock) -> None:
        """Reject logits instead of silently presenting them as probabilities."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.0, 2.0]])
        mock_model_for.return_value = mock_model

        with pytest.raises(RuntimeError, match="must be between zero and one"):
            Naam.pred_rel(["Test Name"])


class TestNaamModelLoading:
    """Test model loading functionality."""

    @patch("tf_keras.models.load_model")
    @patch.object(Naam, "load_model_data")
    def test_model_loading_english(
        self, mock_load_data: Mock, mock_load_model: Mock
    ) -> None:
        """Test loading English model."""
        mock_load_data.return_value = Path("/fake/path")
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        result = Naam._load_model("eng")

        # Check that correct model path was used
        fake_path = Path("/fake/path")
        expected_path = str(fake_path / "eng_and_hindi_models_v2" / "eng_model.keras")
        mock_load_model.assert_called_once_with(expected_path)
        assert result is mock_model

    @patch("tf_keras.models.load_model")
    @patch.object(Naam, "load_model_data")
    def test_model_loading_hindi(
        self, mock_load_data: Mock, mock_load_model: Mock
    ) -> None:
        """Test loading Hindi model."""
        mock_load_data.return_value = Path("/fake/path")
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        result = Naam._load_model("hin")

        fake_path = Path("/fake/path")
        expected_path = str(fake_path / "eng_and_hindi_models_v2" / "hin_model.keras")
        mock_load_model.assert_called_once_with(expected_path)
        assert result is mock_model

    @patch.object(Naam, "load_model_data")
    def test_model_loading_failure_no_data(self, mock_load_data: Mock) -> None:
        """Test model loading failure when data loading fails."""
        mock_load_data.return_value = None

        with pytest.raises(RuntimeError, match="Failed to load model data"):
            Naam._load_model("eng")

    @patch("tf_keras.models.load_model")
    @patch.object(Naam, "load_model_data")
    def test_model_loading_failure_model_error(
        self, mock_load_data: Mock, mock_load_model: Mock
    ) -> None:
        """Test model loading failure when TensorFlow fails."""
        mock_load_data.return_value = Path("/fake/path")
        mock_load_model.side_effect = Exception("TensorFlow error")

        with pytest.raises(RuntimeError, match="Failed to load eng model"):
            Naam._load_model("eng")


class TestNaamLanguageHandling:
    """Test language-specific functionality."""

    @patch.object(Naam, "_load_model")
    def test_models_are_cached_by_language(self, mock_load_model: Mock) -> None:
        """Load each language once and keep both models available."""
        english_model = Mock()
        hindi_model = Mock()
        mock_load_model.side_effect = [english_model, hindi_model]

        assert Naam._model_for("eng") is english_model
        assert Naam._model_for("eng") is english_model
        assert Naam._model_for("hin") is hindi_model
        assert Naam._model_for("hin") is hindi_model

        assert mock_load_model.call_args_list == [
            call("eng", False),
            call("hin", False),
        ]

    @patch.object(Naam, "_load_model")
    def test_latest_replaces_only_requested_language(
        self, mock_load_model: Mock
    ) -> None:
        """Force-refresh one cached language without evicting the other."""
        old_english_model = Mock()
        hindi_model = Mock()
        new_english_model = Mock()
        Naam._models.update(eng=old_english_model, hin=hindi_model)
        mock_load_model.return_value = new_english_model

        assert Naam._model_for("eng", latest=True) is new_english_model
        assert Naam._model_for("hin") is hindi_model
        mock_load_model.assert_called_once_with("eng", True)

    @patch.object(Naam, "_load_model")
    def test_concurrent_requests_load_one_shared_model(
        self, mock_load_model: Mock
    ) -> None:
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
    def test_concurrent_languages_keep_their_own_models(
        self, mock_load_model: Mock
    ) -> None:
        """A Hindi load cannot redirect an in-flight English prediction."""
        english_predicting = Event()
        hindi_loaded = Event()
        english_model = Mock()
        hindi_model = Mock()

        def predict_english(names: list[str], verbose: int) -> np.ndarray:
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
        assert english_result.iloc[0]["pred_prob_muslim"] == 95.0
        assert hindi_result.iloc[0]["pred_label"] == "not-muslim"
        assert hindi_result.iloc[0]["pred_prob_muslim"] == 10.0


class TestNaamErrorHandling:
    """Test error handling in predictions."""

    @patch.object(Naam, "_model_for")
    def test_prediction_with_no_model(self, mock_model_for: Mock) -> None:
        """Test prediction fails when model loading fails."""
        mock_model_for.side_effect = RuntimeError("Model not loaded properly")

        with pytest.raises(RuntimeError, match="Model not loaded properly"):
            Naam.pred_rel(["Test"])

    @patch.object(Naam, "_model_for")
    def test_prediction_tensorflow_error(self, mock_model_for: Mock) -> None:
        """Test handling of TensorFlow prediction errors."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("TensorFlow error")
        mock_model_for.return_value = mock_model

        with pytest.raises(RuntimeError, match="Prediction failed"):
            Naam.pred_rel(["Test"])


class TestNaamIntegration:
    """Integration tests (require actual model download - marked as slow)."""

    # Removed real prediction tests - they were trying to download models and do actual predictions
