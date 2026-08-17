"""Verified access to the immutable model release on Hugging Face."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

MODEL_REPO_ID: Final[str] = "gojiberries/pranaam"
MODEL_REVISION: Final[str] = "96f9cd59cdfe91a8f73f4c72677ec6eeec891a1f"
MODEL_SHA256: Final[dict[str, str]] = {
    "eng/model.safetensors": (
        "b95746bbcb55431b8ea668f2cfdb018caf4c1c67cd914511b4a822d508774f65"
    ),
    "eng/metadata.json": (
        "0548a2e15dec8477c3f6bea0ee44c55e21d561cc6ab7a6096e1483c0e8243d01"
    ),
    "hin/model.safetensors": (
        "d147664f071b2bd336b10cbf9172f6b80efa14d7da9870f68ac1665640fc016e"
    ),
    "hin/metadata.json": (
        "802bbccbd33f1210bc2d3d7680c82c18e2de7378ffa8c65839521d3d8d1e472d"
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
