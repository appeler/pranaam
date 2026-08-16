"""Tests for shared model artifact access."""

from pathlib import Path
from unittest.mock import Mock, patch

from pranaam.base import Base


@patch("pranaam.base.download_model_file")
def test_load_model_data_returns_downloaded_path(mock_download: Mock) -> None:
    """The base wrapper returns the verified Hub cache path."""
    expected = Path("/cache/eng/model.safetensors")
    mock_download.return_value = expected

    assert Base.load_model_data("eng/model.safetensors") == expected
    mock_download.assert_called_once_with("eng/model.safetensors", force_download=False)


@patch("pranaam.base.download_model_file")
def test_latest_forces_cache_refresh(mock_download: Mock) -> None:
    """The public refresh flag reaches the Hub download layer."""
    mock_download.return_value = Path("/cache/model")

    Base.load_model_data("hin/model.safetensors", latest=True)

    mock_download.assert_called_once_with("hin/model.safetensors", force_download=True)
