"""Utilities for downloading and extracting model files."""

import os
import tarfile
from http import HTTPStatus
from pathlib import Path
from typing import Final

import requests
from tqdm.auto import tqdm

from .logging import get_logger

logger = get_logger()

REPO_BASE_URL: Final[str] = (
    os.environ.get("PRANAAM_MODEL_URL")
    or "https://dataverse.harvard.edu/api/access/datafile/13228210"
)


def _challenge_hint(response: requests.Response) -> str:
    """Explain a bot-challenge response, which otherwise looks like a success.

    Args:
        response: The response to the model download request.

    Returns:
        A sentence naming the challenge, or a generic pointer.
    """
    if response.headers.get("x-amzn-waf-action"):
        return (
            "The host is challenging automated clients (x-amzn-waf-action), so "
            "the model cannot be fetched from CI or a datacentre IP. Set "
            "PRANAAM_MODEL_URL to a mirror."
        )
    return "Set PRANAAM_MODEL_URL to a mirror if the host is unreachable."


def download_file(url: str, target: str, file_name: str) -> bool:
    """Download and extract a model file from the given URL.

    Args:
        url: Base URL (not currently used, uses REPO_BASE_URL instead)
        target: Target directory for extraction
        file_name: Name of the file to download

    Returns:
        bool: True if download and extraction successful, False otherwise
    """
    target_path = Path(target)
    file_path = target_path / f"{file_name}.tar.gz"
    try:
        logger.info("Downloading models from dataverse...")

        with (
            requests.Session() as session,
            tqdm(
                unit="iB",
                unit_scale=True,
                desc=file_name,
                ascii=True,
                colour="cyan",
            ) as pbar,
            file_path.open("wb") as file_handle,
        ):
            response = session.get(
                REPO_BASE_URL, stream=True, allow_redirects=True, timeout=120
            )
            response.raise_for_status()
            # raise_for_status() only fires on 4xx/5xx. Dataverse now sits
            # behind an AWS WAF that answers automated clients with 202 and an
            # empty body, which sails past it -- so the check has to be for the
            # response we actually want, not merely the absence of an error.
            if response.status_code != HTTPStatus.OK:
                logger.error(
                    f"Model download refused: {REPO_BASE_URL} returned "
                    f"{response.status_code}, not 200. "
                    f"{_challenge_hint(response)}"
                )
                return False
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None
            pbar.total = total_size

            for chunk in response.iter_content(chunk_size=1024**2):
                if chunk:
                    size = file_handle.write(chunk)
                    pbar.update(size)
        if not file_path.exists():
            logger.error(f"Downloaded file not found at {file_path}")
            return False

        logger.info(f"Downloaded file size: {file_path.stat().st_size} bytes")

        _safe_extract_tar(file_path, target_path)
        file_path.unlink()
        logger.info("Finished downloading models")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading models: {e}")
        return False
    except tarfile.TarError as e:
        # A body that arrived but is not a tarball: an HTML error page, a
        # truncated transfer, or a challenge that answered 200 with something
        # other than the model. "File extraction error: empty file" sent the
        # last person reading this log looking at tarfile, which is the one
        # part that was working.
        logger.error(
            f"File extraction error: {e}. {REPO_BASE_URL} answered, but not "
            f"with a readable tar archive."
        )
        return False
    except OSError as e:
        logger.error(f"File extraction error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading models: {e}")
        return False


def _safe_extract_tar(tar_path: Path, extract_to: Path) -> None:
    """Safely extract tar file preventing path traversal attacks.

    Args:
        tar_path: Path to the tar file
        extract_to: Directory to extract to

    Raises:
        SecurityError: If path traversal attempt detected
    """

    def is_within_directory(directory: Path, target: Path) -> bool:
        abs_directory = directory.resolve()
        abs_target = target.resolve()
        try:
            abs_target.relative_to(abs_directory)
            return True
        except ValueError:
            return False

    with tarfile.open(tar_path, "r:gz") as tar_file:
        for member in tar_file.getmembers():
            member_path = extract_to / member.name
            if not is_within_directory(extract_to, member_path):
                raise SecurityError(
                    f"Attempted path traversal in tar file: {member.name}"
                )

        tar_file.extractall(extract_to)


class SecurityError(Exception):
    """Raised when a security violation is detected."""

    pass
