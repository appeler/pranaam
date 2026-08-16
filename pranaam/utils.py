"""Verified access to the immutable model release on Hugging Face."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

MODEL_REPO_ID: Final[str] = "gojiberries/pranaam"
MODEL_REVISION: Final[str] = "6b92b6a5a5d8abe69f0cd10429dd48e079c00e5f"
MODEL_SHA256: Final[dict[str, str]] = {
    "eng/model.safetensors": (
        "c3b84bd87966f826d44ff594bdfb9dbd4c7ce7e59ae251651a7634c1931f41f7"
    ),
    "eng/vocabulary.txt": (
        "a12fa985bb21c4202332821f05d8d1497f202954e71c3999c9a3532eccc68332"
    ),
    "hin/model.safetensors": (
        "3d31954c85115e37dd9cebcc7654d6fb2d11f69e52ac7a67922544b974d1421b"
    ),
    "hin/vocabulary.txt": (
        "9dceef46783307a364f9e440c3f14bd4b2ed99688cc43ac3e0a73f9a3ebbcd37"
    ),
}


class ModelDownloadError(RuntimeError):
    """Raised when a required model artifact cannot be obtained."""


class ModelIntegrityError(RuntimeError):
    """Raised when model bytes do not match the pinned release manifest."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_file(path: Path, filename: str) -> None:
    if not path.is_file():
        raise ModelIntegrityError(f"Model artifact is missing: {filename}")
    expected = MODEL_SHA256[filename]
    actual = file_sha256(path)
    if actual != expected:
        raise ModelIntegrityError(
            f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
        )


def download_model_file(
    filename: str,
    *,
    force_download: bool = False,
    local_files_only: bool = False,
) -> Path:
    """Resolve and verify one file from the pinned Hugging Face revision.

    Set ``PRANAAM_MODEL_DIR`` to a directory with the published repository
    layout to run from an explicitly managed local mirror.
    """
    if filename not in MODEL_SHA256:
        raise ValueError(f"Unknown model artifact: {filename}")

    local_model_dir = os.environ.get("PRANAAM_MODEL_DIR")
    if local_model_dir:
        path = Path(local_model_dir) / filename
        _verify_model_file(path, filename)
        return path

    def fetch(force: bool) -> Path:
        try:
            return Path(
                hf_hub_download(
                    repo_id=MODEL_REPO_ID,
                    filename=filename,
                    revision=MODEL_REVISION,
                    force_download=force,
                    local_files_only=local_files_only,
                )
            )
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError, ValueError) as exc:
            raise ModelDownloadError(
                f"Could not download {filename} from {MODEL_REPO_ID}@{MODEL_REVISION}"
            ) from exc

    path = fetch(force_download)
    try:
        _verify_model_file(path, filename)
    except ModelIntegrityError:
        if force_download or local_files_only:
            raise
        path = fetch(True)
        _verify_model_file(path, filename)
    return path
