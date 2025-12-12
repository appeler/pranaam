"""Comprehensive tests for naam module."""

from pathlib import Path
from unittest.mock import Mock, patch

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

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_single_string_input(self, mock_model: Mock, mock_load_model: Mock) -> None:
        """Test handling of single string input."""
        # Setup mock model
        mock_model.predict.return_value = np.array([[0.2, 0.8]])
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        result = Naam.pred_rel("Test Name")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Test Name"
        mock_model.predict.assert_called_once_with(["Test Name"], verbose=0)

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_list_input(self, mock_model: Mock, mock_load_model: Mock) -> None:
        """Test handling of list input."""
        mock_model.predict.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        names = ["Name One", "Name Two"]
        result = Naam.pred_rel(names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["name"]) == names

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_pandas_series_input(self, mock_model: Mock, mock_load_model: Mock) -> None:
        """Test handling of pandas Series input."""
        mock_model.predict.return_value = np.array([[0.2, 0.8]])
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        names = pd.Series(["Test Name"])
        result = Naam.pred_rel(names)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1


class TestNaamPredictions:
    """Test prediction functionality."""

    @patch.object(Naam, "_load_model")
    def test_prediction_output_structure(self, mock_load_model: Mock) -> None:
        """Test that predictions return correct DataFrame structure."""
        # Mock the model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        Naam.model = mock_model
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        names = ["Name One", "Name Two"]
        result = Naam.pred_rel(names)

        # Check DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["name", "pred_label", "pred_prob_muslim"]
        assert len(result) == 2

        # Check data types
        assert result["name"].dtype == object
        assert result["pred_label"].dtype == object
        assert pd.api.types.is_numeric_dtype(result["pred_prob_muslim"])

    @patch.object(Naam, "_load_model")
    def test_prediction_labels(self, mock_load_model: Mock) -> None:
        """Test that predictions use correct labels."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        Naam.model = mock_model
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        result = Naam.pred_rel(["Name One", "Name Two"])

        # First prediction: higher muslim probability -> "muslim"
        # Second prediction: higher not-muslim probability -> "not-muslim"
        assert result.iloc[0]["pred_label"] == "muslim"
        assert result.iloc[1]["pred_label"] == "not-muslim"

    @patch.object(Naam, "_load_model")
    def test_prediction_probabilities(self, mock_load_model: Mock) -> None:
        """Test probability calculations."""
        mock_model = Mock()
        # Mock raw logits that will become [0.3, 0.7] after softmax
        # Using logits that softmax to approximately [0.4, 0.6] (60% muslim)
        mock_model.predict.return_value = np.array([[0.0, 0.405]])
        Naam.model = mock_model
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        result = Naam.pred_rel(["Test Name"])

        # Probability should be rounded percentage of muslim class after softmax
        # softmax([0.0, 0.405]) ≈ [0.4, 0.6] -> 60% muslim
        expected_prob = 60.0
        assert result.iloc[0]["pred_prob_muslim"] == expected_prob


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

        # Reset class state
        Naam.weights_loaded = False
        Naam.model = None

        Naam._load_model("eng")

        # Check that correct model path was used
        expected_path = "/fake/path/eng_and_hindi_models_v2/eng_model.keras"
        mock_load_model.assert_called_once_with(expected_path)
        assert Naam.model == mock_model
        assert Naam.weights_loaded is True
        assert Naam.cur_lang == "eng"

    @patch("tf_keras.models.load_model")
    @patch.object(Naam, "load_model_data")
    def test_model_loading_hindi(
        self, mock_load_data: Mock, mock_load_model: Mock
    ) -> None:
        """Test loading Hindi model."""
        mock_load_data.return_value = Path("/fake/path")
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        Naam.weights_loaded = False
        Naam.model = None

        Naam._load_model("hin")

        expected_path = "/fake/path/eng_and_hindi_models_v2/hin_model.keras"
        mock_load_model.assert_called_once_with(expected_path)
        assert Naam.cur_lang == "hin"

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
    @patch.object(Naam, "model")
    def test_language_change_triggers_reload(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test that changing language triggers model reload."""
        mock_model.predict.return_value = np.array([[0.5, 0.5]])

        # Set initial state
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        # Call with different language
        Naam.pred_rel(["Test"], lang="hin")

        # Should trigger model reload
        mock_load_model.assert_called_once_with("hin", False)

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_same_language_no_reload(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test that same language doesn't trigger reload."""
        mock_model.predict.return_value = np.array([[0.5, 0.5]])

        # Set state
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        # Call with same language
        Naam.pred_rel(["Test"], lang="eng")

        # Should not trigger model reload
        mock_load_model.assert_not_called()


class TestNaamErrorHandling:
    """Test error handling in predictions."""

    @patch.object(Naam, "_load_model")
    def test_prediction_with_no_model(self, mock_load_model: Mock) -> None:
        """Test prediction fails when model is None."""
        Naam.model = None
        Naam.weights_loaded = True

        with pytest.raises(RuntimeError, match="Model not loaded properly"):
            Naam.pred_rel(["Test"])

    @patch.object(Naam, "_load_model")
    def test_prediction_tensorflow_error(self, mock_load_model: Mock) -> None:
        """Test handling of TensorFlow prediction errors."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("TensorFlow error")
        Naam.model = mock_model
        Naam.weights_loaded = True

        with pytest.raises(RuntimeError, match="Prediction failed"):
            Naam.pred_rel(["Test"])


class TestNaamIntegration:
    """Integration tests (require actual model download - marked as slow)."""

    # Removed real prediction tests - they were trying to download models and do actual predictions
