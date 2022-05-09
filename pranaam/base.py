import os
from pkg_resources import resource_filename
from .utils import download_file, REPO_BASE_URL
from .logging import get_logger

logger = get_logger()


class Base(object):
    MODELFN = None

    @classmethod
    def load_model_data(cls, file_name, latest=False):
        model_path = None
        if cls.MODELFN:
            model_fn = resource_filename(__name__, cls.MODELFN)
            print(f"Model path {model_fn}")
            if not os.path.exists(model_fn):
                os.makedirs(f"{model_fn}")
            if not os.path.exists(f"{model_fn}/{file_name}") or latest:
                logger.debug(
                    "Downloading model data from the server (this is done only first time) ({0!s})...".format(model_fn)
                )
                if not download_file(REPO_BASE_URL, f"{model_fn}", file_name):
                    logger.error("ERROR: Cannot download model data file")
            else:
                logger.debug("Using model data from {0!s}...".format(model_fn))
            model_path = model_fn

        return model_path
