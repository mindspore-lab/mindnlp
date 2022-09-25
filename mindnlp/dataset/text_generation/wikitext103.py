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
WikiText103 load function
"""
# pylint: disable=C0103

import os
import re
import zipfile
from typing import Union, Tuple
from mindspore.dataset import TextFileDataset
from mindnlp.utils.download import cache_file

URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"

MD5 = "9ddaacaf6af0710eda8c456decff7832"


def unzip(file_path: str, untar_path: str):
    r"""
    Unzip zip file

    Args:
        file_path (str): The path where the zip file is located.
        multiple (str): The directory where the files were unzipped.

    Returns:
        - **names** (list) -All filenames in the zip file.

    Raises:
        TypeError: If `file_path` is not a string.
        TypeError: If `untar_path` is not a string.

    Examples:
        >>> file_path = "./mindnlp/datasets/WikiText103/wikitext-103-v1.zip"
        >>> untar_path = "./mindnlp/datasets/WikiText103"
        >>> output = unzip(file_path,untar_path)
        >>> print(output)
        ['wikitext-103']

    """
    zip_obj = zipfile.ZipFile(file_path)
    names = zip_obj.namelist()
    for name in names:
        zip_obj.extract(name, untar_path)
    zip_obj.close()
    return names

DEFAULT_ROOT = os.path.join(os.path.expanduser('~'), ".mindnlp")

def WikiText103(root: str = DEFAULT_ROOT,
                split: Union[Tuple[str], str] = ('train', 'valid', 'test')):
    r"""
    Load the WikiText103 dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'valid', 'test').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'valid', 'test')
        >>> dataset_train, dataset_valid, dataset_test = WikiText103(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= ' ')]
        [Tensor(shape=[], dtype=String, value= ' = Valkyria Chronicles III = ')]

    """
    cache_dir = os.path.join(root, "datasets", "WikiText103")

    datasets_list = []

    file_path, _ = cache_file(None, cache_dir=cache_dir,
                              url=URL, md5sum=MD5)
    textdir_name = unzip(file_path, os.path.dirname(file_path))
    files_names = os.listdir(os.path.join(cache_dir, textdir_name[0]))
    if isinstance(split, str):
        split = split.split()
    for s in split:
        for filename in files_names:
            if re.search(s, filename):
                dataset = TextFileDataset(os.path.join(
                    cache_dir, textdir_name[0], filename), shuffle=False)
                datasets_list.append(dataset)
    if len(datasets_list) == 1:
        return datasets_list[0]
    return datasets_list
