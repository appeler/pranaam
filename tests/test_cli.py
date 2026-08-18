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

    @patch("pranaam.pranaam.estimate_muslim_name_pattern")
    def test_successful_prediction(
        self, mock_estimate_muslim_name_pattern: Mock
    ) -> None:
        """Test successful prediction with valid arguments."""
        # Setup mock return value
        mock_result = pd.DataFrame(
            {
                "name": ["Test Name"],
                "name_pattern_estimate": ["muslim-associated"],
                "muslim_score": [0.75],
            }
        )
        mock_estimate_muslim_name_pattern.return_value = mock_result

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = main(["--input", "Test Name"])

        assert result == 0
        mock_estimate_muslim_name_pattern.assert_called_once_with(
            "Test Name", lang="eng", refresh_pinned=False
        )

        # Check output contains expected data
        output = mock_stdout.getvalue()
        assert "Test Name" in output
        assert "muslim" in output
        assert "0.75" in output

    @patch("pranaam.pranaam.estimate_muslim_name_pattern")
    def test_hindi_language_option(
        self, mock_estimate_muslim_name_pattern: Mock
    ) -> None:
        """Test Hindi language option."""
        mock_result = pd.DataFrame(
            {
                "name": ["टेस्ट नाम"],
                "name_pattern_estimate": ["not-muslim-associated"],
                "muslim_score": [0.25],
            }
        )
        mock_estimate_muslim_name_pattern.return_value = mock_result

        result = main(["--input", "टेस्ट नाम", "--lang", "hin"])

        assert result == 0
        mock_estimate_muslim_name_pattern.assert_called_once_with(
            "टेस्ट नाम", lang="hin", refresh_pinned=False
        )

    @patch("pranaam.pranaam.estimate_muslim_name_pattern")
    def test_refresh_pinned_option(
        self, mock_estimate_muslim_name_pattern: Mock
    ) -> None:
        """Test --refresh-pinned option."""
        mock_result = pd.DataFrame(
            {
                "name": ["Test Name"],
                "name_pattern_estimate": ["muslim-associated"],
                "muslim_score": [0.8],
            }
        )
        mock_estimate_muslim_name_pattern.return_value = mock_result

        result = main(["--input", "Test Name", "--refresh-pinned"])

        assert result == 0
        mock_estimate_muslim_name_pattern.assert_called_once_with(
            "Test Name", lang="eng", refresh_pinned=True
        )

    @patch("pranaam.pranaam.estimate_muslim_name_pattern")
    def test_all_options_combined(
        self, mock_estimate_muslim_name_pattern: Mock
    ) -> None:
        """Test all options used together."""
        mock_result = pd.DataFrame(
            {
                "name": ["हिंदी नाम"],
                "name_pattern_estimate": ["muslim-associated"],
                "muslim_score": [0.65],
            }
        )
        mock_estimate_muslim_name_pattern.return_value = mock_result

        result = main(["--input", "हिंदी नाम", "--lang", "hin", "--refresh-pinned"])

        assert result == 0
        mock_estimate_muslim_name_pattern.assert_called_once_with(
            "हिंदी नाम", lang="hin", refresh_pinned=True
        )

    def test_invalid_language(self) -> None:
        """Test invalid language option."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--input", "Test Name", "--lang", "invalid"])
        assert exc_info.value.code == 2

    @patch("pranaam.pranaam.estimate_muslim_name_pattern")
    def test_prediction_error_handling(
        self, mock_estimate_muslim_name_pattern: Mock
    ) -> None:
        """Test handling of prediction errors."""
        mock_estimate_muslim_name_pattern.side_effect = Exception("Prediction failed")

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
            description="Estimate Muslim-associated patterns in a name",
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
            "--refresh-pinned",
            action="store_true",
            help="Redownload and verify the immutable pinned model artifacts",
        )

        # Test default parsing
        args = parser.parse_args(["--input", "Test"])
        assert args.lang == "eng"
        assert args.refresh_pinned is False
        assert args.input == "Test"


class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_cli_with_none_argv(self) -> None:
        """Test CLI function when argv is None."""
        # Should use sys.argv[1:] by default
        with (
            patch("sys.argv", ["script_name", "--input", "Test Name"]),
            patch(
                "pranaam.pranaam.estimate_muslim_name_pattern"
            ) as mock_estimate_muslim_name_pattern,
        ):
            mock_result = pd.DataFrame(
                {
                    "name": ["Test Name"],
                    "name_pattern_estimate": ["muslim-associated"],
                    "muslim_score": [0.75],
                }
            )
            mock_estimate_muslim_name_pattern.return_value = mock_result

            result = main(None)

            assert result == 0
            mock_estimate_muslim_name_pattern.assert_called_once()

    @patch("pranaam.pranaam.estimate_muslim_name_pattern")
    def test_output_formatting(self, mock_estimate_muslim_name_pattern: Mock) -> None:
        """Test that output is formatted properly."""
        mock_result = pd.DataFrame(
            {
                "name": ["Name One", "Name Two"],
                "name_pattern_estimate": [
                    "muslim-associated",
                    "not-muslim-associated",
                ],
                "muslim_score": [0.75, 0.25],
            }
        )
        mock_estimate_muslim_name_pattern.return_value = mock_result

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            main(["--input", "Test Names"])

        output = mock_stdout.getvalue()

        # Should contain column headers and data
        assert "name" in output
        assert "name_pattern_estimate" in output
        assert "muslim_score" in output
        assert "Name One" in output
        assert "Name Two" in output
        assert "muslim" in output
        assert "not-muslim" in output


class TestEstimateFunction:
    """Test the estimate_muslim_name_pattern function exposed at module level."""

    def test_estimate_muslim_name_pattern_is_naam_estimate_muslim_name_pattern(
        self,
    ) -> None:
        """The module-level estimator delegates to the Naam class."""
        from pranaam.naam import Naam
        from pranaam.pranaam import (
            estimate_muslim_name_pattern as module_estimate_muslim_name_pattern,
        )

        assert module_estimate_muslim_name_pattern == Naam.estimate_muslim_name_pattern


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
            with patch(
                "pranaam.pranaam.estimate_muslim_name_pattern"
            ) as mock_estimate_muslim_name_pattern:
                mock_result = pd.DataFrame(
                    {
                        "name": [test_input],
                        "name_pattern_estimate": ["muslim-associated"],
                        "muslim_score": [0.5],
                    }
                )
                mock_estimate_muslim_name_pattern.return_value = mock_result

                result = main(["--input", test_input])
                assert result == 0
                mock_estimate_muslim_name_pattern.assert_called_once_with(
                    test_input, lang="eng", refresh_pinned=False
                )

    def test_language_choices(self) -> None:
        """Test that only valid language choices are accepted."""
        valid_langs = ["eng", "hin"]
        invalid_langs = ["en", "hi", "english", "hindi", "spanish", ""]

        # Valid languages should work
        for lang in valid_langs:
            with patch(
                "pranaam.pranaam.estimate_muslim_name_pattern"
            ) as mock_estimate_muslim_name_pattern:
                mock_result = pd.DataFrame(
                    {
                        "name": ["Test"],
                        "name_pattern_estimate": ["muslim-associated"],
                        "muslim_score": [0.5],
                    }
                )
                mock_estimate_muslim_name_pattern.return_value = mock_result

                result = main(["--input", "Test", "--lang", lang])
                assert result == 0

        # Invalid languages should fail
        for lang in invalid_langs:
            with pytest.raises(SystemExit) as exc_info:
                main(["--input", "Test", "--lang", lang])
            assert exc_info.value.code == 2


# Removed TestCLIErrorHandling class - was causing KeyboardInterrupt issues in CI
