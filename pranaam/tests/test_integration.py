"""Integration tests for pranaam package."""

import pytest
import pandas as pd
from unittest.mock import patch, Mock
from typing import Any
import numpy as np

from pranaam import pred_rel, Naam


class TestPackageIntegration:
    """Test package-level integration."""

    def test_package_imports(self) -> None:
        """Test that all expected symbols are importable."""
        from pranaam import pred_rel, Naam

        assert pred_rel is not None
        assert Naam is not None
        assert callable(pred_rel)
        assert hasattr(Naam, "pred_rel")

    def test_pred_rel_is_naam_method(self) -> None:
        """Test that package pred_rel is the Naam class method."""
        assert pred_rel == Naam.pred_rel

    def test_package_version(self) -> None:
        """Test that package has version information."""
        import pranaam

        assert hasattr(pranaam, "__version__")
        assert pranaam.__version__ == "0.0.2"

    def test_package_all_exports(self) -> None:
        """Test __all__ exports are correct."""
        import pranaam

        assert hasattr(pranaam, "__all__")
        assert "pred_rel" in pranaam.__all__
        assert "Naam" in pranaam.__all__


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_single_name_workflow(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test complete workflow with single name."""
        # Setup mock model
        mock_model.predict.return_value = np.array([[0.2, 0.8]])
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        # Test workflow
        result = pred_rel("Shah Rukh Khan")

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["name", "pred_label", "pred_prob_muslim"]

        # Verify result content
        assert result.iloc[0]["name"] == "Shah Rukh Khan"
        assert result.iloc[0]["pred_label"] == "muslim"
        assert isinstance(result.iloc[0]["pred_prob_muslim"], (int, float))

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_batch_names_workflow(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test complete workflow with multiple names."""
        # Setup mock model for batch prediction
        mock_model.predict.return_value = np.array(
            [
                [0.2, 0.8],  # muslim
                [0.7, 0.3],  # not-muslim
                [0.1, 0.9],  # muslim
            ]
        )
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        names = ["Shah Rukh Khan", "Amitabh Bachchan", "Mohammed Ali"]
        result = pred_rel(names)

        # Verify batch processing
        assert len(result) == 3
        assert list(result["name"]) == names
        assert result.iloc[0]["pred_label"] == "muslim"
        assert result.iloc[1]["pred_label"] == "not-muslim"
        assert result.iloc[2]["pred_label"] == "muslim"

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_language_switching_workflow(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test workflow with language switching."""
        mock_model.predict.return_value = np.array([[0.3, 0.7]])
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        # First prediction in English
        result1 = pred_rel(["Test Name"], lang="eng")
        assert len(result1) == 1

        # Switch to Hindi - should trigger model reload
        result2 = pred_rel(["टेस्ट नाम"], lang="hin")
        assert len(result2) == 1

        # Verify model was reloaded for Hindi
        mock_load_model.assert_called_with("hin", False)


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_mixed_religion_predictions(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test predictions for names of different religions."""

        # Mock different probabilities for different name types
        def mock_predict(names: Any, verbose: int = 0) -> Any:
            results = []
            for name in names:
                name_lower = name.lower()
                if any(
                    muslim_indicator in name_lower
                    for muslim_indicator in ["mohammed", "khan", "ali", "shah"]
                ):
                    results.append([0.15, 0.85])  # High Muslim probability
                elif any(
                    hindu_indicator in name_lower
                    for hindu_indicator in ["sharma", "bachchan", "gupta"]
                ):
                    results.append([0.85, 0.15])  # Low Muslim probability
                else:
                    results.append([0.5, 0.5])  # Neutral
            return np.array(results)

        mock_model.predict = mock_predict
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        test_names = [
            "Mohammed Khan",  # Should predict Muslim
            "Amitabh Bachchan",  # Should predict not-Muslim
            "John Smith",  # Neutral
            "Ali Ahmed",  # Should predict Muslim
        ]

        result = pred_rel(test_names)

        assert len(result) == 4
        assert result.iloc[0]["pred_label"] == "muslim"  # Mohammed Khan
        assert result.iloc[1]["pred_label"] == "not-muslim"  # Amitabh Bachchan
        assert result.iloc[2]["pred_label"] in [
            "muslim",
            "not-muslim",
        ]  # John Smith (neutral)
        assert result.iloc[3]["pred_label"] == "muslim"  # Ali Ahmed

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_edge_case_names(self, mock_model: Mock, mock_load_model: Mock) -> None:
        """Test with edge case names."""
        mock_model.predict.return_value = np.array(
            [
                [0.6, 0.4],  # not-muslim
                [0.3, 0.7],  # muslim
                [0.5, 0.5],  # neutral
            ]
        )
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        edge_names = [
            "A",  # Single character
            "Very Long Name With Many Words",  # Long name
            "Name123",  # With numbers
        ]

        result = pred_rel(edge_names)

        assert len(result) == 3
        assert all(isinstance(name, str) for name in result["name"])
        assert all(label in ["muslim", "not-muslim"] for label in result["pred_label"])
        assert all(0 <= prob <= 100 for prob in result["pred_prob_muslim"])

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_probability_ranges(self, mock_model: Mock, mock_load_model: Mock) -> None:
        """Test that probabilities are in expected ranges."""
        # Test various probability values
        test_probs = [
            [0.0, 1.0],  # 100% Muslim
            [1.0, 0.0],  # 0% Muslim
            [0.5, 0.5],  # 50% Muslim
            [0.25, 0.75],  # 75% Muslim
            [0.9, 0.1],  # 10% Muslim
        ]

        mock_model.predict.return_value = np.array(test_probs)
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        names = [f"Name{i}" for i in range(5)]
        result = pred_rel(names)

        # Check probability conversion
        expected_muslim_probs = [100, 0, 50, 75, 10]  # Rounded percentages

        for i, expected_prob in enumerate(expected_muslim_probs):
            actual_prob = result.iloc[i]["pred_prob_muslim"]
            assert (
                abs(actual_prob - expected_prob) <= 1
            )  # Allow for rounding differences


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    def test_invalid_input_recovery(self) -> None:
        """Test that invalid inputs provide clear error messages."""
        with pytest.raises(ValueError, match="Unsupported language"):
            pred_rel(["Test"], lang="invalid")

        with pytest.raises(ValueError, match="Input names list cannot be empty"):
            pred_rel([])

    @patch.object(Naam, "_load_model")
    def test_model_loading_failure_recovery(self, mock_load_model: Mock) -> None:
        """Test recovery from model loading failures."""
        mock_load_model.side_effect = RuntimeError("Model loading failed")
        Naam.weights_loaded = False

        with pytest.raises(RuntimeError, match="Model loading failed"):
            pred_rel(["Test Name"])

    @patch.object(Naam, "_load_model")
    def test_prediction_failure_recovery(self, mock_load_model: Mock) -> None:
        """Test recovery from prediction failures."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("TensorFlow error")
        Naam.model = mock_model
        Naam.weights_loaded = True

        with pytest.raises(RuntimeError, match="Prediction failed"):
            pred_rel(["Test Name"])


class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_batch_processing_efficiency(
        self, mock_model: Mock, mock_load_model: Mock
    ) -> None:
        """Test that batch processing is more efficient than individual calls."""
        call_count = 0

        def count_predict_calls(names: Any, verbose: int = 0) -> Any:
            nonlocal call_count
            call_count += 1
            return np.array([[0.5, 0.5]] * len(names))

        mock_model.predict = count_predict_calls
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        # Batch processing
        call_count = 0
        names = ["Name1", "Name2", "Name3", "Name4", "Name5"]
        pred_rel(names)
        batch_calls = call_count

        # Individual processing
        call_count = 0
        for name in names:
            pred_rel([name])
        individual_calls = call_count

        # Batch should be more efficient (fewer model calls)
        assert batch_calls < individual_calls
        assert batch_calls == 1  # Should be single batch call
        assert individual_calls == 5  # Should be 5 individual calls

    @patch.object(Naam, "_load_model")
    @patch.object(Naam, "model")
    def test_model_caching(self, mock_model: Mock, mock_load_model: Mock) -> None:
        """Test that model is cached and not reloaded unnecessarily."""
        mock_model.predict.return_value = np.array([[0.5, 0.5]])
        Naam.weights_loaded = True
        Naam.cur_lang = "eng"

        # Multiple predictions with same language
        pred_rel(["Name1"])
        pred_rel(["Name2"])
        pred_rel(["Name3"])

        # Model should not be reloaded
        mock_load_model.assert_not_called()

        # Verify model.predict was called multiple times
        assert mock_model.predict.call_count == 3
