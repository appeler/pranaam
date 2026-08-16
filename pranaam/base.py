"""Shared access to versioned model artifacts."""

from pathlib import Path

from .utils import download_model_file


class Base:
    """Base class for loading files from the pinned model release."""

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> Path:
        """Return a verified local path for a released model artifact."""
        return download_model_file(file_name, force_download=latest)
