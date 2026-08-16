"""
End-to-end integration tests with real models and predictions.
These tests actually download models and run real predictions.
"""

from collections.abc import Generator

import pandas as pd
import pytest

import pranaam
from pranaam.naam import Naam


class TestRealModelDownloadAndPrediction:
    """Test real model download and prediction functionality."""

    @pytest.fixture(autouse=True)
    def setup_clean_environment(self) -> Generator[None, None, None]:
        """Ensure clean model state for each test."""
        Naam._models.clear()
        yield
        Naam._models.clear()

    @pytest.mark.integration
    def test_real_english_predictions(self) -> None:
        """Test real predictions with English names using actual models."""
        # Real Bollywood actor names with expected patterns
        test_names = [
            "Shah Rukh Khan",  # Expected: Muslim
            "Salman Khan",  # Expected: Muslim
            "Aamir Khan",  # Expected: Muslim
            "Saif Ali Khan",  # Expected: Muslim
            "Amitabh Bachchan",  # Expected: Not Muslim
            "Akshay Kumar",  # Expected: Not Muslim
            "Hrithik Roshan",  # Expected: Not Muslim
        ]

        # This will download models if not cached
        result = pranaam.pred_rel(test_names, lang="eng")

        # Verify DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["name", "pred_label", "pred_prob_muslim"]
        assert len(result) == len(test_names)

        # Verify all names are present
        assert set(result["name"]) == set(test_names)

        # Verify prediction labels are valid
        valid_labels = {"muslim", "not-muslim"}
        assert all(label in valid_labels for label in result["pred_label"])

        # Verify probabilities are reasonable (0-100)
        assert all(0 <= prob <= 100 for prob in result["pred_prob_muslim"])

        # Verify expected patterns (these are actual predictions, not mocks)
        khan_results = result[result["name"].str.contains("Khan")]
        muslim_khans = khan_results[khan_results["pred_label"] == "muslim"]

        # Should predict most Khans as Muslim (this is what the model should do)
        assert len(muslim_khans) >= 3, (
            f"Expected at least 3 Khans predicted as Muslim, got {len(muslim_khans)}"
        )

    @pytest.mark.integration
    def test_real_hindi_predictions(self) -> None:
        """Test real predictions with Hindi names using actual models."""
        # Real Hindi names in Devanagari
        test_names = [
            "शाहरुख खान",  # Shah Rukh Khan
            "सलमान खान",  # Salman Khan
            "अमिताभ बच्चन",  # Amitabh Bachchan
            "अक्षय कुमार",  # Akshay Kumar
        ]

        # This will download Hindi models if not cached
        result = pranaam.pred_rel(test_names, lang="hin")

        # Verify DataFrame structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["name", "pred_label", "pred_prob_muslim"]
        assert len(result) == len(test_names)

        # Verify all names are present
        assert set(result["name"]) == set(test_names)

        # Verify prediction labels are valid
        valid_labels = {"muslim", "not-muslim"}
        assert all(label in valid_labels for label in result["pred_label"])

        # Verify probabilities are reasonable
        assert all(0 <= prob <= 100 for prob in result["pred_prob_muslim"])

    @pytest.mark.integration
    def test_model_caching_behavior(self) -> None:
        """Test that models are properly cached after first download."""
        # First prediction - should trigger download
        result1 = pranaam.pred_rel("Shah Rukh Khan", lang="eng")
        assert "eng" in Naam._models

        # Second prediction - should use cached model
        result2 = pranaam.pred_rel("Amitabh Bachchan", lang="eng")
        assert list(Naam._models) == ["eng"]

        # Results should be consistent
        assert result1.columns.tolist() == result2.columns.tolist()

    @pytest.mark.integration
    def test_language_switching(self) -> None:
        """Test switching between English and Hindi models."""
        # Start with English
        eng_result = pranaam.pred_rel("Shah Rukh Khan", lang="eng")
        assert set(Naam._models) == {"eng"}

        # Switch to Hindi - should reload model
        hin_result = pranaam.pred_rel("शाहरुख खान", lang="hin")
        assert set(Naam._models) == {"eng", "hin"}

        # Switch back to English - should reload model again
        eng_result2 = pranaam.pred_rel("Salman Khan", lang="eng")
        assert set(Naam._models) == {"eng", "hin"}

        # All results should have same structure
        for result in [eng_result, hin_result, eng_result2]:
            assert list(result.columns) == ["name", "pred_label", "pred_prob_muslim"]

    @pytest.mark.integration
    def test_pandas_series_integration(self) -> None:
        """Test real pandas Series input integration."""
        # Create DataFrame with real names
        df = pd.DataFrame(
            {
                "actor_name": ["Shah Rukh Khan", "Amitabh Bachchan", "Salman Khan"],
                "movie_count": [50, 100, 45],
            }
        )

        # Use pandas Series as input
        result = pranaam.pred_rel(df["actor_name"], lang="eng")

        # Verify integration
        assert len(result) == len(df)
        assert all(name in df["actor_name"].values for name in result["name"])

        # Create combined result
        combined = pd.concat([df, result[["pred_label", "pred_prob_muslim"]]], axis=1)

        # Verify combined structure
        expected_cols = ["actor_name", "movie_count", "pred_label", "pred_prob_muslim"]
        assert list(combined.columns) == expected_cols

    @pytest.mark.integration
    def test_batch_processing_performance(self) -> None:
        """Test batch processing with real models."""
        import time

        # Large batch of real names
        names = [
            "Shah Rukh Khan",
            "Amitabh Bachchan",
            "Salman Khan",
            "Aamir Khan",
            "Akshay Kumar",
            "Hrithik Roshan",
            "Ranbir Kapoor",
            "Saif Ali Khan",
            "Ajay Devgan",
            "John Abraham",
            "Arjun Kapoor",
            "Varun Dhawan",
        ]

        start_time = time.time()
        result = pranaam.pred_rel(names, lang="eng")
        end_time = time.time()

        processing_time = end_time - start_time
        # Verify all names processed
        assert len(result) == len(names)
        assert set(result["name"]) == set(names)

        # Performance should be reasonable (adjust based on actual performance)
        assert processing_time < 10.0, (
            f"Batch processing too slow: {processing_time:.2f}s"
        )

    @pytest.mark.integration
    def test_error_handling_real_scenarios(self) -> None:
        """Test error handling in real scenarios."""
        # Test with empty input
        with pytest.raises(ValueError, match="empty"):
            pranaam.pred_rel("", lang="eng")

        # Test with invalid language
        with pytest.raises(ValueError, match="Unsupported language"):
            pranaam.pred_rel("Test Name", lang="invalid")  # type: ignore[arg-type]

        # Test with None input
        with pytest.raises((ValueError, TypeError)):
            pranaam.pred_rel(None, lang="eng")  # type: ignore[arg-type]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.integration
    def test_mixed_cultural_names(self) -> None:
        """Test predictions on mixed cultural background names."""
        mixed_names = [
            "John Smith",  # Western
            "Mohammed Ali",  # Arabic/Muslim
            "Priya Sharma",  # Hindu
            "David Johnson",  # Western
            "Fatima Khan",  # Muslim
            "Raj Patel",  # Hindu
        ]

        result = pranaam.pred_rel(mixed_names, lang="eng")

        # Verify structure
        assert len(result) == len(mixed_names)
        assert all(0 <= prob <= 100 for prob in result["pred_prob_muslim"])

    @pytest.mark.integration
    def test_edge_case_names(self) -> None:
        """Test edge cases with real models."""
        edge_cases = [
            "A",  # Single character
            "Mohammad",  # Common Muslim name variant
            "Krishna",  # Hindu deity name
            "Ali Khan",  # Short Muslim name
            "Ram Singh",  # Hindu name
        ]

        result = pranaam.pred_rel(edge_cases, lang="eng")

        assert len(result) == len(edge_cases)


# Marker for running only E2E tests
pytestmark = pytest.mark.integration
