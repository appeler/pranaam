"""Shared model-data loading support."""

from pathlib import Path
from typing import ClassVar

from filelock import FileLock, Timeout
from platformdirs import user_cache_path

from .logging import get_logger
from .utils import SecurityError, download_file, get_model_url, verify_model_files

logger = get_logger()


class Base:
    """Base class for model data management and loading."""

    MODELFN: ClassVar[str | None] = None

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> Path | None:
        """Return a verified model directory, downloading it when needed.

        Args:
            file_name: Name of the model bundle directory
            latest: Whether to refresh an existing cached bundle

        Returns:
            The model cache directory, or ``None`` when no usable bundle exists.
        """
        if cls.MODELFN is None:
            return None

        model_dir = Path(user_cache_path("pranaam", ensure_exists=True)) / cls.MODELFN
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / file_name
        lock = FileLock(model_dir / f".{file_name}.lock", timeout=600)

        try:
            with lock:
                cache_is_verified = False
                if target.is_dir():
                    try:
                        verify_model_files(target)
                        cache_is_verified = True
                    except (SecurityError, OSError) as error:
                        logger.warning(
                            "Ignoring invalid model cache at %s: %s", target, error
                        )

                if cache_is_verified and not latest:
                    logger.debug("Using model data from %s", model_dir)
                    return model_dir

                logger.debug("Downloading model data to %s", model_dir)
                if download_file(get_model_url(), str(model_dir), file_name):
                    if target.is_dir():
                        return model_dir
                    logger.error("Downloaded model bundle is missing %s", target)
                    return None

                if cache_is_verified and target.is_dir():
                    logger.warning(
                        "Model refresh failed; continuing with the verified cache at %s",
                        target,
                    )
                    return model_dir

                logger.error("Cannot download model data")
                return None
        except Timeout:
            logger.error(
                "Timed out waiting for the model cache lock at %s", lock.lock_file
            )
            return None
