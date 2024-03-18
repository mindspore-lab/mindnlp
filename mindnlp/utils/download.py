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
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=simplifiable-if-expression
# pylint: disable=missing-function-docstring
# pylint: disable=too-many-return-statements
# pylint: disable=unspecified-encoding
# pylint: disable=try-except-raise
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
import tempfile
import time
from typing import Union, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from tqdm.autonotebook import tqdm
import requests
from requests.exceptions import ProxyError, SSLError, HTTPError

from mindnlp.configs import DEFAULT_ROOT, ENV_VARS_TRUE_VALUES, MINDNLP_CACHE, REPO_TYPES, HF_URL_BASE, \
    HF_TOKEN
from .errors import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    ModelNotFoundError,
    GatedRepoError
)
from . import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_CACHED_NO_EXIST = object()
_CACHED_NO_EXIST_T = Any

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False

def is_offline_mode():
    return _is_offline_mode

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
    for deprecated behavior allowing to download config/models with a single url instead of using the Hub.

    Args:
        url (`str`): The url of the file to download.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

    Returns:
        `str`: The location of the temporary file where the url was downloaded.
    """
    return http_get(url, tempfile.gettempdir(), download_file_name='tmp_' + url.split('/')[-1], proxies=proxies)

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
ca
    """
    if "CACHE_DIR" in os.environ:
        cache_dir = os.environ.get("CACHE_DIR")
        if os.path.isdir(cache_dir):
            return cache_dir
        raise NotADirectoryError(
            f"{os.environ['CACHE_DIR']} is not a directory.")
    cache_dir = DEFAULT_ROOT

    return cache_dir


def http_get(url, path=None, md5sum=None, download_file_name=None, proxies=None, headers=None):
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

    # subfolder
    if '/' in name and not os.path.exists(file_path[:file_path.rfind('/')]):
        os.makedirs(file_path[:file_path.rfind('/')])

    while not (os.path.exists(file_path) and check_md5(file_path, md5sum)):
        if retry_cnt < retry_limit:
            retry_cnt += 1
        else:
            raise HTTPError(
                f"Download from {url} failed. " "Retry limit reached")
        req = requests.get(url, stream=True, timeout=10, proxies=proxies, headers=headers)

        status = req.status_code
        if status == 404:
            raise EntryNotFoundError(f"Can not found url: {url}")
        if status == 401:
            raise GatedRepoError('You should have authorization to access the model.')
        if status == 429:
            raise HTTPError('Too many requests.')
        try:
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
        except requests.exceptions.RequestException as e:
            if retry_cnt >= retry_limit:
                raise
            print(f"Failed to download: {e}")
            print(f"Retrying... (attempt {retry_cnt}/{retry_limit})")
            time.sleep(1)  # Add a small delay before retrying

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

def get_file_from_repo(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo) or `None` if the
        file does not exist.

    Examples:

    ```python
    # Download a tokenizer configuration from huggingface.co and cache.
    tokenizer_config = get_file_from_repo("google-bert/bert-base-uncased", "tokenizer_config.json")
    # This model does not have a tokenizer config so the result will be None.
    tokenizer_config = get_file_from_repo("FacebookAI/xlm-roberta-base", "tokenizer_config.json")
    ```
    """
    return cached_file(
        path_or_repo_id=path_or_repo,
        filename=filename,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        subfolder=subfolder,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
    )



def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    local_files_only: bool = False,
    revision = 'main',
    token = None,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_gated_repo: bool = True,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```"""
    # Private arguments
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}."
                )
            return None
        return resolved_file

    if cache_dir is None:
        cache_dir = MINDNLP_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not force_download:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, repo_type=repo_type
        )
        if resolved_file is not None:
            if resolved_file is not object():
                return resolved_file
            if not _raise_exceptions_for_missing_entries:
                return None
            raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")
    try:
        # Load from URL or cache if already cached
        resolved_file = download(
            path_or_repo_id,
            filename,
            subfolder=None if len(subfolder) == 0 else subfolder,
            repo_type=repo_type,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            revision=revision,
            token=token
        )
    except GatedRepoError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        if resolved_file is not None or not _raise_exceptions_for_gated_repo:
            return resolved_file
        raise EnvironmentError(
            "You are trying to access a gated repo.\nMake sure to have access to it at "
            f"https://huggingface.co/{path_or_repo_id}.\n{str(e)}"
        ) from e
    except RepositoryNotFoundError as e:
        raise EnvironmentError(
            f"{path_or_repo_id} is not a local folder and is nost a valid model identifier "
        ) from e
    except LocalEntryNotFoundError as e:
        # We try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir)
        if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
            return resolved_file
        if not _raise_exceptions_for_missing_entries or not _raise_exceptions_for_connection_errors:
            return None
        raise EnvironmentError(
            f"We couldn't load this file, couldn't find it in the"
            f" cached files and it looks like {path_or_repo_id} is not the path to a directory containing a file named"
            f" {full_filename}.\nCheckout your internet connection or see how to run the library in offline mode at"
        ) from e
    except EntryNotFoundError as e:
        if not _raise_exceptions_for_missing_entries:
            return None
        raise EnvironmentError(
            f"{path_or_repo_id} does not appear to have a file named {full_filename}."
        ) from e

    except HTTPError as err:
        # First we try to see if we have a cached version (not up to date):
        resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir)
        if resolved_file is not None and resolved_file != object():
            return resolved_file
        if not _raise_exceptions_for_connection_errors:
            return None

        raise EnvironmentError(f"There was a specific connection error when trying to load {path_or_repo_id}:\n{err}") from err

    return resolved_file


def download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    resume_download: bool = False,
    local_files_only: bool = False,
    revision: str = 'main',
    token: str = None,
) -> str:
    """Download a given file if it's not already present in the local cache.
    """
    if cache_dir is None:
        cache_dir = MINDNLP_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(local_dir, Path):
        local_dir = str(local_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")

    storage_folder = os.path.join(cache_dir, repo_type, repo_id)
    os.makedirs(storage_folder, exist_ok=True)

    # cross platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}' on Windows. Please ask the repository"
                " owner to rename this file."
            )

    pointer_path = os.path.join(storage_folder, relative_filename)

    if os.path.exists(pointer_path) and not force_download:
        return pointer_path

    url = build_download_url(repo_id, filename, revision, repo_type=repo_type)
    # check model whether exist
    model_url = url[: url.rfind(repo_id) + len(repo_id)]

    token = HF_TOKEN if not token else token

    headers = None
    if token:
        headers = {
            'authorization': f"Bearer {token}"
        }

    try:
        req = requests.head(model_url, timeout=3, proxies=proxies, headers=headers)
        if req.status_code >= 400:
            raise RepositoryNotFoundError(f"Can not found model: {repo_id}")
        pointer_path = http_get(url, storage_folder, download_file_name=relative_filename, proxies=proxies, headers=headers)
    except (requests.exceptions.SSLError,
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout):
        # Otherwise, our Internet connection is down.
        # etag is None
        raise

    return pointer_path

# https://modelscope.cn/api/v1/models/mindnlp/THUDM_chatglm-6b/repo?Revision=master&FilePath=mindspore-00001-of-00008.ckpt

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
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Union[str, _CACHED_NO_EXIST_T, None]:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo on huggingface.co.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to `"main"` if it's not provided and no `commit_hash` is
            provided either.
        repo_type (`str`, *optional*):
            The type of the repository. Will default to `"model"`.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.

    Example:

    ```python
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    filepath = try_to_load_from_cache()
    if isinstance(filepath, str):
        # file exists and is cached
        ...
    elif filepath is _CACHED_NO_EXIST:
        # non-existence of file is cached
        ...
    else:
        # file is not cached
        ...
    ```
    """
    if revision is None:
        revision = "main"
    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")
    if cache_dir is None:
        cache_dir = MINDNLP_CACHE

    repo_cache = os.path.join(cache_dir, f"{repo_type}/{repo_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None

    # Check if file exists in cache
    cache_file = os.path.join(repo_cache, filename)
    return cache_file if os.path.isfile(cache_file) else None


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    revision='main',
    token=None,
    user_agent=None,
    subfolder="",
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
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir
    )
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # Load from URL
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision,
                token=token
            )
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        except EntryNotFoundError as exc:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            ) from exc
        except HTTPError as exc:
            raise EnvironmentError(
                f"We couldn't load {shard_filename}. You should try"
                " again after checking your internet connection."
            ) from exc

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata


def build_download_url(
    repo_id: str,
    filename: str,
    revision: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> str:
    """Construct the URL of a file from the given information.
    """
    return HF_URL_BASE.format(repo_id, revision, filename)
