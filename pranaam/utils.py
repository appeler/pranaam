# -*- coding: utf-8 -*-

import os
import gdown
from .logging import get_logger

logger = get_logger()

REPO_BASE_URL = (
    os.environ.get("PRANAAM_MODEL_URL") or "https://drive.google.com/drive/folders/17OlN960ZH4cud3r-mSq846iQoOGbwkcl"
)


def download_file(url, target):
    status = True
    try:
        print("Download models ...")
        gdown.download_folder(url, quiet=True, output=target)
        print("Finished downloading models ...")
    except Exception as exe:
        logger.error(f"Not able to download models {exe}")
        status = False
    return status
