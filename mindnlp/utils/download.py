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
Download functions
"""

import os
import shutil
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse
import requests
from tqdm.autonotebook import tqdm
from mindnlp.configs import DEFAULT_ROOT

def get_cache_path():
    r"""
    Get the storage path of the default cache. If the environment 'cache_path' is set, use the environment variable.

    Args:
        None

    Returns:
        str, the path of default or the environment 'cache_path'.

    Examples:
        >>> default_cache_path = get_cache_path()
        >>> print(default_cache_path)
        '{home}\.mindnlp'

    """
    if "CACHE_DIR" in os.environ:
        cache_dir = os.environ.get("CACHE_DIR")
        if os.path.isdir(cache_dir):
            return cache_dir
        raise NotADirectoryError(
            f"{os.environ['CACHE_DIR']} is not a directory.")
    cache_dir = DEFAULT_ROOT

    return cache_dir


def http_get(url, path=None, md5sum=None, download_file_name=None, proxies=None):
    r"""
    Download from given url, save to path.

    Args:
        url (str): download url
        path (str): download to given path (default value: '{home}\.text')
        md5sum (str): The true md5sum of download file.
        download_file_name(str): The name of the downloaded file.\
            (This para meter is required if the end of the link is not the downloaded file name.)
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        str, the path of default or the environment 'cache_path'.

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
    name = ""
    if download_file_name is None:
        name = os.path.split(url)[-1]
    else:
        name = download_file_name

    filename = os.path.join(path, name)

    while not (os.path.exists(filename) and check_md5(filename, md5sum)):
        if retry_cnt < retry_limit:
            retry_cnt += 1
        else:
            raise RuntimeError(
                f"Download from {url} failed. " "Retry limit reached")

        req = requests.get(url, stream=True, timeout=10, proxies=proxies)
        if req.status_code != 200:
            raise RuntimeError(
                f"Downloading from {url} failed with code {req.status_code}!"
            )

        tmp_filename = filename + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_filename, "wb") as file:
            if total_size:
                with tqdm(
                    total=int(total_size), unit="B", unit_scale=True, unit_divisor=1024
                ) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        file.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
        shutil.move(tmp_filename, filename)

    return Path(path), filename


def check_md5(filename: str, md5sum=None):
    r"""
    Check md5 of download file.

    Args:
        filename (str): The fullname of download file.
        md5sum (str): The true md5sum of download file.

    Returns:
        bool, the md5 check result.

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
    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            md5.update(chunk)
    md5hex = md5.hexdigest()

    if md5hex != md5sum:
        return False
    return True


def get_filepath(path: str):
    r"""
    Get the filepath of file.

    Args:
        path (str): The path of the required file.

    Returns:
        - str, If `path` is a folder containing a file, return `{path}\{filename}`;
          if `path` is a folder containing multiple files or a single file, return `path`.

    Raises:
        TypeError: If `path` is not a string.
        RuntimeError: If `path` is None.

    Examples:
        >>> path = '{home}\.text'
        >>> get_filepath_result = get_filepath(path)
        >>> print(get_filepath_result)
        '{home}\.text'

    """
    if os.path.isdir(path):
        files = os.listdir(path)
        if len(files) == 1:
            return os.path.join(path, files[0])
        return path
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"{path} is not a valid file or directory.")


def cache_file(
    filename: str,
    cache_dir: str = None,
    url: str = None,
    md5sum=None,
    download_file_name=None,
    proxies=None,
):
    r"""
    If there is the file in cache_dir, return the path; if there is no such file, use the url to download.

    Args:
        filename (str): The name of the required dataset file.
        cache_dir (str): The path of save the file.
        url (str): The url of the required dataset file.
        md5sum (str): The true md5sum of download file.
        download_file_name(str): The name of the downloaded file.\
            (This parameter is required if the end of the link is not the downloaded file name.)
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - str, If `path` is a folder containing a file, return `{path}\{filename}`;
          if `path` is a folder containing multiple files or a single file, return `path`.

    Raises:
        TypeError: If `filename` is not a string.
        TypeError: If `cache_dir` is not a string.
        TypeError: If `url` is not a string.
        RuntimeError: If `filename` is None.

    Examples:
        >>> filename = 'aclImdb_v1'
        >>> path, filename = cache_file(filename)
        >>> print(path, filename)
        '{home}\.text' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        cache_dir = get_cache_path()

    path, filename = cached_path(
        filename_or_url=url,
        cache_dir=cache_dir,
        folder_name=None,
        md5sum=md5sum,
        download_file_name=download_file_name,
        proxies=proxies,
    )

    return path, filename


def cached_path(
    filename_or_url: str,
    cache_dir: str = None,
    folder_name=None,
    md5sum=None,
    download_file_name=None,
    proxies=None,
):
    r"""
    If there is the file in cache_dir, return the path; if there is no such file, use the url to download.

    Args:
        filename_or_url (str): The name or url of the required file .
        cache_dir (str): The path of save the file.
        folder_name (str): The additional folder to which the dataset is cached.(under the `cache_dir`)
        md5sum (str): The true md5sum of download file.
        download_file_name(str): The name of the downloaded file.\
            (This parameter is required if the end of the link is not the downloaded file name.)
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - str, If `path` is a folder containing a file, return `{path}\{filename}`;
          if `path` is a folder containing multiple files or a single file, return `path`.

    Raises:
        TypeError: If `path` is not a string.
        RuntimeError: If `path` is None.

    Examples:
        >>> path = "https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz"
        >>> path, filename = cached_path(path)
        >>> print(path, filename)
        '{home}\.text\aclImdb_v1.tar.gz' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        dataset_cache = Path(get_cache_path())
    else:
        dataset_cache = cache_dir

    if folder_name:
        dataset_cache = os.path.join(dataset_cache, folder_name)

    parsed = urlparse(filename_or_url)

    if parsed.scheme in ("http", "https"):
        return get_from_cache(
            filename_or_url,
            Path(dataset_cache),
            md5sum=md5sum,
            download_file_name=download_file_name,
            proxies=proxies,
        )
    if (
        parsed.scheme == ""
        and Path(os.path.join(dataset_cache, filename_or_url)).exists()
    ):
        return Path(os.path.join(dataset_cache, filename_or_url))
    if parsed.scheme == "":
        raise FileNotFoundError(
            f"file {filename_or_url} not found in {dataset_cache}.")
    raise ValueError(
        f"unable to parse {filename_or_url} as a URL or as a local path")


def match_file(filename: str, cache_dir: str) -> str:
    r"""
    If there is the file in cache_dir, return the path; otherwise, return empty string or error.

    Args:
        filename (str): The name of the required file.
        cache_dir (str): The path of save the file.

    Returns:
        - str, If there is the file in cache_dir, return filename;
          if there is no such file, return empty string '';
          if there are two or more matching file, report an error.

    Raises:
        TypeError: If `filename` is not a string.
        TypeError: If `cache_dir` is not a string.
        RuntimeError: If `filename` is None.
        RuntimeError: If `cache_dir` is None.

    Examples:
        >>> name = 'aclImdb_v1.tar.gz'
        >>> path = get_cache_path()
        >>> match_file_result = match_file(name, path)

    """
    files = os.listdir(cache_dir)
    matched_filenames = []
    for file_name in files:
        if re.match(filename + "$", file_name) or re.match(
            filename + "\\..*", file_name
        ):
            matched_filenames.append(file_name)
    if not matched_filenames:
        return ""
    if len(matched_filenames) == 1:
        return matched_filenames[-1]
    raise RuntimeError(
        f"Duplicate matched files:{matched_filenames}, this should be caused by a bug."
    )


def get_from_cache(
    url: str, cache_dir: str = None, md5sum=None, download_file_name=None, proxies=None
):
    r"""
    If there is the file in cache_dir, return the path; if there is no such file, use the url to download.

    Args:
        url (str): The path to download the file.
        cache_dir (str): The path of save the file.
        md5sum (str): The true md5sum of download file.
        download_file_name(str): The name of the downloaded file.\
            (This parameter is required if the end of the link is not the downloaded file name.)
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - str, The path of save the downloaded file.
        - str, The name of downloaded file.

    Raises:
        TypeError: If `url` is not a string.
        TypeError: If `cache_dir` is not a Path.
        RuntimeError: If `url` is None.

    Examples:
        >>> path = "https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/aclImdb_v1.tar.gz"
        >>> path, filename = cached_path(path)
        >>> print(path, filename)
        '{home}\.text' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        cache_dir = Path(get_cache_path())
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = ""
    if download_file_name is None:
        filename = re.sub(r".+/", "", url)
    else:
        filename = download_file_name

    match_dir_name = match_file(filename, cache_dir)
    dir_name = filename
    if match_dir_name:
        dir_name = match_dir_name
    cache_path = cache_dir / dir_name
    if cache_path.exists() and check_md5(cache_path, md5sum):
        return get_filepath(cache_path), filename
    path = http_get(url, cache_dir, md5sum,
                    download_file_name=download_file_name, proxies=proxies)[1]
    return Path(path), filename
