from importlib.resources import files
from pathlib import Path

from .logging import get_logger
from .utils import REPO_BASE_URL, download_file

logger = get_logger()


class Base:
    """Base class for model data management and loading."""

    MODELFN: str | None = None

    @classmethod
    def load_model_data(cls, file_name: str, latest: bool = False) -> Path | None:
        """Load model data, downloading if necessary.

        Args:
            file_name: Name of the model file to load
            latest: Whether to force download of latest version

        Returns:
            Path | None: Path to the model directory, or None if loading failed
        """
        model_path: Path | None = None
        if cls.MODELFN:
            # Use modern importlib.resources instead of deprecated pkg_resources
            package_dir = files(__package__)
            model_dir = Path(str(package_dir)) / cls.MODELFN
            model_dir.mkdir(exist_ok=True)

            target_file = model_dir / file_name
            if not target_file.exists() or latest:
                logger.debug(
                    f"Downloading model data from the server (this is done only first time) ({model_dir})..."
                )
                if not download_file(REPO_BASE_URL, str(model_dir), file_name):
                    logger.error("ERROR: Cannot download model data file")
            else:
                logger.debug(f"Using model data from {model_dir}...")
            model_path = model_dir

        return model_path
