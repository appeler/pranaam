import os
from pkg_resources import resource_filename
from .utils import download_file, REPO_BASE_URL
from .logging import get_logger

logger = get_logger()


class Base(object):
    MODELFN = None

    @classmethod
    def load_model_data(cls, latest=False):
        model_path = None
        if cls.MODELFN:
            model_fn = resource_filename(__name__, cls.MODELFN)
            path = os.path.dirname(model_fn)
            print(f"Model path {path}")
            if not os.path.exists(path):
                os.makedirs(path)
            if not os.path.exists(f"{model_fn}/saved_model") or latest:
                logger.debug("Downloading model data from the server ({0!s})...".format(model_fn))
                if not download_file(REPO_BASE_URL, f"{model_fn}/saved_model"):
                    logger.error("ERROR: Cannot download model data file")
            else:
                logger.debug("Using model data from {0!s}...".format(model_fn))
            model_path = model_fn

        return model_path
