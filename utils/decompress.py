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
Decompress functions
"""

import os
import tarfile
import zipfile
import gzip

def untar(file_path: str, untar_path: str):
    r"""
    Untar tar.gz file

    Args:
        file_path (str): The path where the tgz file is located.
        multiple (str): The directory where the files were unzipped.

    Returns:
        - **names** (list) -All filenames in the tar.gz file.

    Raises:
        TypeError: If `file_path` is not a string.
        TypeError: If `untar_path` is not a string.

    Examples:
        >>> file_path = "./mindnlp/datasets/IWSLT2016/2016-01.tgz"
        >>> untar_path = "./mindnlp/datasets/IWSLT2016"
        >>> output = untar(file_path,untar_path)
        >>> print(output[0])
        '2016-01'

    """
    tar = tarfile.open(file_path)
    names = tar.getnames()
    for name in names:
        if os.path.exists(os.path.join(untar_path, name)):
            continue
        tar.extract(name, untar_path)
    tar.close()
    return names


def unzip(file_path: str, unzip_path: str):
    r"""
    Untar .zip file

    Args:
        file_path (str): The path where the .zip file is located.
        unzip_path (str): The directory where the files were unzipped.

    Returns:
        - **names** (list) -All filenames in the .zip file.

    Raises:
        TypeError: If `file_path` is not a string.
        TypeError: If `untar_path` is not a string.

    """
    zipf = zipfile.ZipFile(file_path, "r")
    for name in zipf.namelist():
        zipf.extract(name, unzip_path)
    zipf.close()
    return zipf.namelist()

def ungz(file_path: str, unzip_path: str = None):
    r"""
    Untar .gz file

    Args:
        file_path (str): The path where the .gz file is located.
        unzip_path (str): The directory where the files were unzipped.

    Returns:
        - **unzip_path** (str): The directory where the files were unzipped.

    Raises:
        TypeError: If `file_path` is not a string.
        TypeError: If `untar_path` is not a string.

    """
    if not isinstance(unzip_path,str):
        unzip_path = str(file_path)[:-3]
    with open(unzip_path,'wb') as file:
        gz_file = gzip.open(file_path, mode = 'rb')
        file.write(gz_file.read())
        gz_file.close()
    return unzip_path
