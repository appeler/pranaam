"""Utilities for downloading and installing model bundles."""

import hashlib
import os
import shutil
import tarfile
import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Final

import requests
from tqdm.auto import tqdm

from .logging import get_logger

logger = get_logger()

DEFAULT_MODEL_URL: Final[str] = (
    "https://dataverse.harvard.edu/api/access/datafile/13228210"
)
MODEL_SHA256: Final[dict[str, str]] = {
    "eng_model.keras": (
        "2cdf998ede5d71715e3cfa084adc81de3b546597fc0257f929bd0e26b2e19202"
    ),
    "hin_model.keras": (
        "e1b14656749bc63198b81275c978cea4959821e00233ebf82a773f1118e575c9"
    ),
}
CHUNK_SIZE: Final[int] = 1024**2


def get_model_url() -> str:
    """Return the configured model-bundle URL."""
    return os.environ.get("PRANAAM_MODEL_URL") or DEFAULT_MODEL_URL


def download_file(url: str, target: str, file_name: str) -> bool:
    """Download, verify, and atomically install a model bundle.

    Args:
        url: URL of the model archive
        target: Target directory for the installed bundle
        file_name: Expected top-level directory in the archive

    Returns:
        ``True`` when a verified bundle was installed, otherwise ``False``.
    """
    target_path = Path(target)
    try:
        target_path.mkdir(parents=True, exist_ok=True)
        _recover_interrupted_install(target_path, file_name)
        with tempfile.TemporaryDirectory(
            prefix=f".{file_name}-", dir=target_path
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            archive_path = temporary_path / "model.tar.gz"
            extracted_path = temporary_path / "extracted"
            extracted_path.mkdir()

            _download_archive(url, archive_path, file_name)
            _safe_extract_tar(archive_path, extracted_path)
            extracted_model = extracted_path / file_name
            verify_model_files(extracted_model)
            _install_verified_model(extracted_model, target_path, file_name)

        logger.info("Installed verified model bundle at %s", target_path / file_name)
        return True
    except requests.exceptions.RequestException as error:
        logger.error("Network error downloading models: %s", error)
    except (SecurityError, tarfile.TarError, OSError, ValueError) as error:
        logger.error("Model bundle validation failed: %s", error)
    except Exception as error:
        logger.error("Unexpected error downloading models: %s", error)
    return False


def _download_archive(url: str, archive_path: Path, file_name: str) -> None:
    """Stream a complete HTTP 200 response into a temporary archive."""
    with (
        requests.Session() as session,
        tqdm(
            unit="iB",
            unit_scale=True,
            desc=file_name,
            ascii=True,
            colour="cyan",
        ) as progress,
        archive_path.open("wb") as archive,
    ):
        response = session.get(url, stream=True, allow_redirects=True, timeout=120)
        response.raise_for_status()
        if response.status_code != HTTPStatus.OK:
            raise requests.HTTPError(
                f"{url} returned {response.status_code}, not 200. "
                f"{_challenge_hint(response)}",
                response=response,
            )

        content_length = response.headers.get("Content-Length")
        expected_size = int(content_length) if content_length else None
        progress.total = expected_size
        downloaded_size = 0

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                written = archive.write(chunk)
                downloaded_size += written
                progress.update(written)

    if downloaded_size == 0:
        raise SecurityError("Downloaded model archive is empty")
    if expected_size is not None and downloaded_size != expected_size:
        raise SecurityError(
            f"Downloaded {downloaded_size} bytes; expected {expected_size}"
        )


def _challenge_hint(response: requests.Response) -> str:
    """Explain a bot-challenge response that otherwise looks successful."""
    if response.headers.get("x-amzn-waf-action"):
        return (
            "The host challenged this client. Set PRANAAM_MODEL_URL to a "
            "trusted mirror."
        )
    return "Set PRANAAM_MODEL_URL to a trusted mirror if this host is unavailable."


def verify_model_files(model_path: Path) -> None:
    """Verify every required model against its pinned SHA-256 digest."""
    if not model_path.is_dir():
        raise SecurityError(f"Downloaded archive is missing {model_path.name}")

    for name, expected_digest in MODEL_SHA256.items():
        path = model_path / name
        if not path.is_file():
            raise SecurityError(f"Downloaded archive is missing {name}")

        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(CHUNK_SIZE), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise SecurityError(f"Checksum mismatch for {name}")


def _recover_interrupted_install(target: Path, file_name: str) -> None:
    """Restore the previous bundle if a process stopped during the directory swap."""
    installed = target / file_name
    backup = target / f".{file_name}.previous"
    if installed.exists():
        _remove_path(backup)
    elif backup.exists():
        os.replace(backup, installed)


def _install_verified_model(extracted: Path, target: Path, file_name: str) -> None:
    """Swap a verified bundle into place and roll back a failed replacement."""
    installed = target / file_name
    backup = target / f".{file_name}.previous"
    _remove_path(backup)

    if installed.exists():
        os.replace(installed, backup)
    try:
        os.replace(extracted, installed)
    except Exception:
        if backup.exists() and not installed.exists():
            os.replace(backup, installed)
        raise
    else:
        _remove_path(backup)


def _remove_path(path: Path) -> None:
    """Remove a file, link, or directory when it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _safe_extract_tar(tar_path: Path, extract_to: Path) -> None:
    """Extract a tar archive without allowing writes outside the destination."""
    extract_root = extract_to.resolve()
    with tarfile.open(tar_path, "r:gz") as tar_file:
        for member in tar_file.getmembers():
            member_path = (extract_to / member.name).resolve()
            try:
                member_path.relative_to(extract_root)
            except ValueError as error:
                raise SecurityError(
                    f"Attempted path traversal in tar file: {member.name}"
                ) from error
        tar_file.extractall(extract_to, filter="data")


class SecurityError(Exception):
    """Raised when model-bundle validation detects unsafe or unexpected data."""
