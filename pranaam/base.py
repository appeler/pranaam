import os
from typing import Optional
from importlib.resources import files

from .utils import download_file, REPO_BASE_URL
from .logging import get_logger

logger = get_logger()


class Base:
    """Base class for model data management and loading."""

    MODELFN: Optional[str] = None

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> Optional[str]:
        """Load model data, downloading if necessary.

        Args:
            file_name: Name of the model file to load
            latest: Whether to force download of latest version

        Returns:
            Path to the model directory, or None if loading failed
        """
        model_path: Optional[str] = None
        if cls.MODELFN:
            # Use modern importlib.resources instead of deprecated pkg_resources
            package_dir = files(__package__)
            model_fn = str(package_dir / cls.MODELFN)
            print(f"Model path {model_fn}")
            if not os.path.exists(model_fn):
                os.makedirs(f"{model_fn}")
            if not os.path.exists(f"{model_fn}/{file_name}") or latest:
                logger.debug(
                    "Downloading model data from the server (this is done only first time) ({0!s})...".format(
                        model_fn
                    )
                )
                if not download_file(REPO_BASE_URL, f"{model_fn}", file_name):
                    logger.error("ERROR: Cannot download model data file")
            else:
                logger.debug("Using model data from {0!s}...".format(model_fn))
            model_path = model_fn

        return model_path
