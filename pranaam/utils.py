"""Utilities for downloading and extracting model files."""

import os
import tarfile
from typing import Optional

import requests
from tqdm.auto import tqdm

from .logging import get_logger

logger = get_logger()

REPO_BASE_URL: str = os.environ.get(
    "PRANAAM_MODEL_URL"
) or "https://dataverse.harvard.edu/api/access/datafile/6286241"


def download_file(url: str, target: str, file_name: str) -> bool:
    """Download and extract a model file from the given URL.
    
    Args:
        url: Base URL (not currently used, uses REPO_BASE_URL instead)
        target: Target directory for extraction
        file_name: Name of the file to download
        
    Returns:
        True if download and extraction successful, False otherwise
    """
    file_path = f"{target}/{file_name}.tar.gz"
    try:
        logger.info("Downloading models from dataverse...")
        
        with requests.Session() as session:
            response = session.get(REPO_BASE_URL, stream=True, allow_redirects=True, timeout=30)
            response.raise_for_status()
        content_length = response.headers.get('Content-Length')
        total_size = int(content_length) if content_length else None
        
        with tqdm(
            total=total_size, 
            unit='iB', 
            unit_scale=True, 
            desc=file_name,
            ascii=True, 
            colour='cyan'
        ) as pbar:
            with open(file_path, 'wb') as file_handle:
                for chunk in response.iter_content(chunk_size=1024**2):
                    if chunk:  # filter out keep-alive chunks
                        size = file_handle.write(chunk)
                        pbar.update(size)
        # Extract tar file with safety checks
        _safe_extract_tar(file_path, target)
        # Clean up downloaded tar file
        os.remove(file_path)
        logger.info("Finished downloading models")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error downloading models: {e}")
        return False
    except (tarfile.TarError, OSError) as e:
        logger.error(f"File extraction error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading models: {e}")
        return False


def _safe_extract_tar(tar_path: str, extract_to: str) -> None:
    """Safely extract tar file preventing path traversal attacks.
    
    Args:
        tar_path: Path to the tar file
        extract_to: Directory to extract to
        
    Raises:
        Exception: If path traversal attempt detected
        tarfile.TarError: If tar file is corrupted
    """
    def is_within_directory(directory: str, target: str) -> bool:
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory

    with tarfile.open(tar_path, "r:gz") as tar_file:
        for member in tar_file.getmembers():
            member_path = os.path.join(extract_to, member.name)
            if not is_within_directory(extract_to, member_path):
                raise Exception(f"Attempted path traversal in tar file: {member.name}")
        
        tar_file.extractall(extract_to)
