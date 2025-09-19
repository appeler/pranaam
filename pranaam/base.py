import os
from importlib.resources import files

from .logging import get_logger
from .utils import REPO_BASE_URL, download_file

logger = get_logger()


class Base:
    """Base class for model data management and loading."""

    MODELFN: str | None = None

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> str | None:
        """Load model data, downloading if necessary.

        Args:
            file_name: Name of the model file to load
            latest: Whether to force download of latest version

        Returns:
            Path to the model directory, or None if loading failed
        """
        model_path: str | None = None
        if cls.MODELFN:
            # Use modern importlib.resources instead of deprecated pkg_resources
            package_dir = files(__package__)
            model_fn = str(package_dir / cls.MODELFN)
            if not os.path.exists(model_fn):
                os.makedirs(f"{model_fn}")
            if not os.path.exists(f"{model_fn}/{file_name}") or latest:
                logger.debug(
                    f"Downloading model data from the server (this is done only first time) ({model_fn!s})..."
                )
                if not download_file(REPO_BASE_URL, f"{model_fn}", file_name):
                    logger.error("ERROR: Cannot download model data file")
            else:
                logger.debug(f"Using model data from {model_fn!s}...")
            model_path = model_fn

        return model_path
