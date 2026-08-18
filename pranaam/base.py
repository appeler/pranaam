"""Shared access to versioned model artifacts."""

from pathlib import Path

from .utils import download_model_file


class Base:
    """Base class for loading files from the pinned model release."""

    @classmethod
    def load_model_data(cls, file_name: str, refresh_pinned: bool = False) -> Path:
        """Return a verified path, optionally refreshing its pinned source."""
        return download_model_file(file_name, force_download=refresh_pinned)
