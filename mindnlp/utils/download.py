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
# pylint: disable=C0103
"""
Download functions
"""

import os
import shutil
import hashlib
import re
import json
import types
import functools
from typing import Union, Optional
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import requests
from tqdm.autonotebook import tqdm
from requests.exceptions import ProxyError, SSLError

from mindnlp.configs import DEFAULT_ROOT
from .errors import ModelNotFoundError


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

def extract_filename_from_url(url):
    """extract filename from url"""
    parsed_url = urlparse(url)

    path_segments = parsed_url.path.split('/')
    file_from_path = path_segments[-1]

    # for modelscope
    query_params = parse_qs(parsed_url.query)
    file_from_query = query_params.get('FilePath', [''])[0]

    return file_from_query if file_from_query else file_from_path


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
    if not os.path.exists(path):
        os.makedirs(path)

    retry_cnt = 0
    retry_limit = 3

    if download_file_name is None:
        name = extract_filename_from_url(url)
    else:
        name = download_file_name

    file_path = os.path.join(path, name)

    while not (os.path.exists(file_path) and check_md5(file_path, md5sum)):
        if retry_cnt < retry_limit:
            retry_cnt += 1
        else:
            raise RuntimeError(
                f"Download from {url} failed. " "Retry limit reached")

        req = requests.get(url, stream=True, timeout=10, proxies=proxies)

        status = req.status_code
        if status == 404:
            raise ModelNotFoundError(f"Can not found url: {url}")

        tmp_file_path = file_path + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_file_path, "wb") as file:
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
        shutil.move(tmp_file_path, file_path)

    return file_path


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


def cached_file(
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
        >>> path, filename = cached_file(filename)
        >>> print(path, filename)
        '{home}\.text' 'aclImdb_v1.tar.gz'

    """
    if cache_dir is None:
        cache_dir = get_cache_path()

    path = cached_path(
        filename_or_url=url,
        cache_dir=cache_dir,
        md5sum=md5sum,
        download_file_name=download_file_name,
        proxies=proxies,
    )

    return path, filename


def cached_path(
    filename_or_url: str,
    cache_dir: str = None,
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
    parsed = urlparse(filename_or_url)

    if parsed.scheme == "":
        if os.path.exists(filename_or_url):
            return filename_or_url

    if cache_dir is None:
        dataset_cache = get_cache_path()
    else:
        dataset_cache = cache_dir

    if (
        parsed.scheme == ""
        and os.path.exists(os.path.join(dataset_cache, filename_or_url))
    ):
        return os.path.join(dataset_cache, filename_or_url)

    if parsed.scheme in ("http", "https"):
        return get_from_cache(
            filename_or_url,
            dataset_cache,
            md5sum=md5sum,
            download_file_name=download_file_name,
            proxies=proxies,
        )
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
        if re.match(filename + "$", file_name):
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
        raise ValueError('cache dir should not be None.')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if download_file_name is None:
        filename = extract_filename_from_url(url)
    else:
        filename = download_file_name

    file_path = os.path.join(cache_dir, filename)

    if os.path.exists(file_path) and check_md5(file_path, md5sum):
        return file_path
    try:
        path = http_get(url, cache_dir, md5sum, download_file_name=filename, proxies=proxies)
        return path
    except (ProxyError, SSLError) as exc:
        raise exc
    except ModelNotFoundError:
        return None

def try_to_load_from_cache(
    filename: str,
    cache_dir: Union[str, Path, None] = None,
) -> Optional[str]:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        filename (`str`):
            The filename to look for inside `repo_id`.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.
    """
    if not os.path.isdir(cache_dir):
        # No cache for this model
        return None

    cached_file_ = os.path.join(cache_dir, filename)

    return cached_file_ if os.path.isfile(cached_file_) else None


def get_checkpoint_shard_files(
    index_filename,
    cache_dir=None,
    url=None,
    force_download=False,
    proxies=None,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {cache_dir}.")

    with open(index_filename, "r", encoding='utf-8') as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(cache_dir):
        shard_filenames = [os.path.join(cache_dir, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(shard_filenames[-1], cache_dir=cache_dir)
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
            # Load from URL
        cached_filename = cached_path(
            '/'.join([url, shard_filename]),
            cache_dir,
            proxies=proxies
        )
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        if cached_filename is None:
            raise EnvironmentError(
                f"{cache_dir} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata
