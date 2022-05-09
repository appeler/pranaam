# -*- coding: utf-8 -*-

import os
import tarfile
import requests
from .logging import get_logger

logger = get_logger()

REPO_BASE_URL = os.environ.get("PRANAAM_MODEL_URL") or "https://dataverse.harvard.edu/api/access/datafile/6286241"


def download_file(url, target, file_name):
    status = True
    file_path = f"{target}/{file_name}.tar.gz"
    try:
        print("Download models from dataverse...")
        # download the file
        r = requests.get(url, allow_redirects=True)
        open(file_path, "wb").write(r.content)
        # untar
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(target)
        # remove zip file
        os.remove(file_path)
        print("Finished downloading models ...")
    except Exception as exe:
        logger.error(f"Not able to download models {exe}")
        status = False
    return status
