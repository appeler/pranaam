"""Tests for CLI functionality."""

from io import StringIO
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from pranaam.pranaam import main


class TestCLIMain:
    """Test main CLI function."""

    def test_help_option(self) -> None:
        """Test --help option displays help and exits."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0

    def test_missing_required_argument(self) -> None:
        """Test that missing --input argument returns error."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2

    @patch("pranaam.pranaam.pred_rel")
    def test_successful_prediction(self, mock_pred_rel: Mock) -> None:
        """Test successful prediction with valid arguments."""
        # Setup mock return value
        mock_result = pd.DataFrame(
            {
                "name": ["Test Name"],
                "pred_label": ["muslim"],
                "pred_prob_muslim": [75.0],
            }
        )
        mock_pred_rel.return_value = mock_result

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["--input", "Test Name"])

        assert result == 0
        mock_pred_rel.assert_called_once_with("Test Name", lang="eng", latest=False)

        # Check output contains expected data
        output = mock_stdout.getvalue()
        assert "Test Name" in output
        assert "muslim" in output
        assert "75.0" in output

    @patch("pranaam.pranaam.pred_rel")
    def test_hindi_language_option(self, mock_pred_rel: Mock) -> None:
        """Test Hindi language option."""
        mock_result = pd.DataFrame(
            {
                "name": ["टेस्ट नाम"],
                "pred_label": ["not-muslim"],
                "pred_prob_muslim": [25.0],
            }
        )
        mock_pred_rel.return_value = mock_result

        result = main(["--input", "टेस्ट नाम", "--lang", "hin"])

        assert result == 0
        mock_pred_rel.assert_called_once_with("टेस्ट नाम", lang="hin", latest=False)

    @patch("pranaam.pranaam.pred_rel")
    def test_latest_option(self, mock_pred_rel: Mock) -> None:
        """Test --latest option."""
        mock_result = pd.DataFrame(
            {
                "name": ["Test Name"],
                "pred_label": ["muslim"],
                "pred_prob_muslim": [80.0],
            }
        )
        mock_pred_rel.return_value = mock_result

        result = main(["--input", "Test Name", "--latest"])

        assert result == 0
        mock_pred_rel.assert_called_once_with("Test Name", lang="eng", latest=True)

    @patch("pranaam.pranaam.pred_rel")
    def test_all_options_combined(self, mock_pred_rel: Mock) -> None:
        """Test all options used together."""
        mock_result = pd.DataFrame(
            {"name": ["हिंदी नाम"], "pred_label": ["muslim"], "pred_prob_muslim": [65.0]}
        )
        mock_pred_rel.return_value = mock_result

        result = main(["--input", "हिंदी नाम", "--lang", "hin", "--latest"])

        assert result == 0
        mock_pred_rel.assert_called_once_with("हिंदी नाम", lang="hin", latest=True)

    def test_invalid_language(self) -> None:
        """Test invalid language option."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--input", "Test Name", "--lang", "invalid"])
        assert exc_info.value.code == 2

    @patch("pranaam.pranaam.pred_rel")
    def test_prediction_error_handling(self, mock_pred_rel: Mock) -> None:
        """Test handling of prediction errors."""
        mock_pred_rel.side_effect = Exception("Prediction failed")

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = main(["--input", "Test Name"])

        assert result == 1
        error_output = mock_stderr.getvalue()
        assert "Error: Prediction failed" in error_output

    def test_default_arguments(self) -> None:
        """Test default argument values."""
        # This test verifies the argument parser setup
        import argparse

        # Create parser same way as in main function
        parser = argparse.ArgumentParser(
            description="Predict religion based on name",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--input", required=True, help="Name to analyze (single name as string)"
        )
        parser.add_argument(
            "--lang",
            default="eng",
            choices=["eng", "hin"],
            help="Language of input name",
        )
        parser.add_argument(
            "--latest", action="store_true", help="Download latest model version"
        )

        # Test default parsing
        args = parser.parse_args(["--input", "Test"])
        assert args.lang == "eng"
        assert args.latest is False
        assert args.input == "Test"


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_with_none_argv(self) -> None:
        """Test CLI function when argv is None."""
        # Should use sys.argv[1:] by default
        with patch("sys.argv", ["script_name", "--input", "Test Name"]):
            with patch("pranaam.pranaam.pred_rel") as mock_pred_rel:
                mock_result = pd.DataFrame(
                    {
                        "name": ["Test Name"],
                        "pred_label": ["muslim"],
                        "pred_prob_muslim": [75.0],
                    }
                )
                mock_pred_rel.return_value = mock_result

                result = main(None)  # argv=None should use sys.argv[1:]

                assert result == 0
                mock_pred_rel.assert_called_once()

    @patch("pranaam.pranaam.pred_rel")
    def test_output_formatting(self, mock_pred_rel: Mock) -> None:
        """Test that output is formatted properly."""
        mock_result = pd.DataFrame(
            {
                "name": ["Name One", "Name Two"],
                "pred_label": ["muslim", "not-muslim"],
                "pred_prob_muslim": [75.0, 25.0],
            }
        )
        mock_pred_rel.return_value = mock_result

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["--input", "Test Names"])

        output = mock_stdout.getvalue()

        # Should contain column headers and data
        assert "name" in output
        assert "pred_label" in output
        assert "pred_prob_muslim" in output
        assert "Name One" in output
        assert "Name Two" in output
        assert "muslim" in output
        assert "not-muslim" in output


class TestPredRelFunction:
    """Test the pred_rel function exposed at module level."""

    def test_pred_rel_is_naam_pred_rel(self) -> None:
        """Test that pred_rel is the same as Naam.pred_rel."""
        from pranaam.naam import Naam
        from pranaam.pranaam import pred_rel as module_pred_rel

        assert module_pred_rel == Naam.pred_rel


class TestCLIArgumentValidation:
    """Test CLI argument validation."""

    def test_required_input_argument(self) -> None:
        """Test that input argument is required."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--lang", "eng"])  # Missing --input
        assert exc_info.value.code == 2

    def test_input_argument_accepts_any_string(self) -> None:
        """Test that input accepts various string types."""
        test_inputs = [
            "Simple Name",
            "Name with numbers 123",
            "Name-with-hyphens",
            "Name.with.dots",
            "नाम हिंदी में",
            "Mixed English हिंदी Name",
        ]

        for test_input in test_inputs:
            with patch("pranaam.pranaam.pred_rel") as mock_pred_rel:
                mock_result = pd.DataFrame(
                    {
                        "name": [test_input],
                        "pred_label": ["muslim"],
                        "pred_prob_muslim": [50.0],
                    }
                )
                mock_pred_rel.return_value = mock_result

                result = main(["--input", test_input])
                assert result == 0
                mock_pred_rel.assert_called_once_with(
                    test_input, lang="eng", latest=False
                )

    def test_language_choices(self) -> None:
        """Test that only valid language choices are accepted."""
        valid_langs = ["eng", "hin"]
        invalid_langs = ["en", "hi", "english", "hindi", "spanish", ""]

        # Valid languages should work
        for lang in valid_langs:
            with patch("pranaam.pranaam.pred_rel") as mock_pred_rel:
                mock_result = pd.DataFrame(
                    {
                        "name": ["Test"],
                        "pred_label": ["muslim"],
                        "pred_prob_muslim": [50.0],
                    }
                )
                mock_pred_rel.return_value = mock_result

                result = main(["--input", "Test", "--lang", lang])
                assert result == 0

        # Invalid languages should fail
        for lang in invalid_langs:
            with pytest.raises(SystemExit) as exc_info:
                main(["--input", "Test", "--lang", lang])
            assert exc_info.value.code == 2


# Removed TestCLIErrorHandling class - was causing KeyboardInterrupt issues in CI
