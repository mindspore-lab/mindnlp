# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Download Functional
"""

import os
import shutil
import hashlib
import json
import requests
from tqdm import tqdm


def get_cache_path():
    r"""
    Get the storage path of the default cache. If the environment 'cache_path' is set, use the environment variable.

    Args:
        None

    Returns:
        - **cache_dir**(str) - The path of default or the environment 'cache_path'.

    Examples:
        >>> default_cache_path = get_cache_path()
        >>> print(default_cache_path)
        '{home}\.text'

    """
    if "CACHE_DIR" in os.environ:
        cache_dir = os.environ.get('CACHE_DIR')
        if os.path.isdir(cache_dir):
            return cache_dir
        raise NotADirectoryError(f"{os.environ['CACHE_DIR']} is not a directory.")
    cache_dir = os.path.expanduser(os.path.join("~", ".text"))

    return cache_dir

def http_get(url, path=None, md5sum=None):
    r"""
    Download from given url, save to path.

    Args:
        url (str): download url
        path (str): download to given path (default value: '{home}\.text')

    Returns:
        - **cache_dir**(str) - The path of default or the environment 'cache_path'.

    Raises:
        TypeError: If `url` is not a String.
        RuntimeError: If `url` is None.

    Examples:
        >>> url = 'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz'
        >>> cache_path = http_get(url)
        >>> print(cache_path)
        ('{home}\.text', '{home}\aclImdb_v1.tar.gz')

    """
    if path is None:
        path = get_cache_path()

    if not os.path.exists(path):
        os.makedirs(path)

    retry_cnt = 0
    retry_limit = 3
    name = os.path.split(url)[-1]
    filename = os.path.join(path, name)

    while not (os.path.exists(filename) and check_md5(filename, md5sum)):
        if retry_cnt < retry_limit:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        tmp_filename = filename + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_filename, 'wb') as f:
            if total_size:
                with tqdm(total=int(total_size),
                          unit='B',
                          unit_scale=True,
                          unit_divisor=1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_filename, filename)

    return path, filename

def check_md5(filename, md5sum=None):
    r"""
    Check md5 of download file.

    Args:
        filename (str) : The fullname of download file.
        md5sum (str) : The true md5sum of download file.

    Returns:
        - ** md5_check_result ** (bool) - The md5 check result.

    Raises:
        TypeError: If `filename` is not a string.
        RuntimeError: If `filename` is None.

    Examples:
        >>> filename = 'test'
        >>> check_md5_result = check_md5(filename)
        True

    """
    if md5sum is None:
        return True

    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    md5hex = md5.hexdigest()

    if md5hex != md5sum:
        return False
    return True

def get_dataset_url(datasetname):
    r"""
    Get dataset url for download

    Args:
        datasetname (str) : The name of the dataset to download.

    Returns:
        - ** url ** (str) - The url of the dataset to download.

    Raises:
        TypeError: If `datasetname` is not a string.
        RuntimeError: If `datasetname` is None.

    Examples:
        >>> name = 'aclImdb_v1'
        >>> print(get_dataset_url(name))
        'https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz'

    """
    default_dataset_json = './text/config/dataset_url.json'
    with open(default_dataset_json, "r") as json_file:
        json_dict = json.load(json_file)

    url = json_dict.get(datasetname, None)
    if url:
        return url
    raise KeyError(f"There is no {datasetname}.")
