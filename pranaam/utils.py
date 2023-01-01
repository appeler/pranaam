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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_ref, target)
        # remove zip file
        os.remove(file_path)
        print("Finished downloading models ...")
    except Exception as exe:
        logger.error(f"Not able to download models {exe}")
        status = False
    return status
