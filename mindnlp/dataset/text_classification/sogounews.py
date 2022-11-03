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
SogouNews dataset
"""
# pylint: disable=C0103

import os
import csv
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar

csv.field_size_limit(500000)

URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUkVqNEszd0pHaFE&confirm=t"

MD5 = "0c1700ba70b73f964dd8de569d3fd03e"


class Sogounews:
    """
    SogouNews dataset source
    """

    def __init__(self, path) -> None:
        self.path: str = path
        self._label, self._text = [], []
        self._load()

    def _load(self):
        csvfile = open(self.path, "r", encoding="utf-8")
        dict_reader = csv.reader(csvfile)
        for row in dict_reader:
            self._label.append(int(row[0]))
            self._text.append(f"{row[1]} {row[2]}")

    def __getitem__(self, index):
        return self._label[index], self._text[index]

    def __len__(self):
        return len(self._label)


@load.register
def SogouNews(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "test"),
    proxies=None,
):
    r"""
    Load the SogouNews dataset

    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'test').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ("train", "test")
        >>> dataset_train,dataset_test = SogouNews(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """

    cache_dir = os.path.join(root, "datasets", "SogouNews")
    path_dict = {
        "train": "train.csv",
        "test": "test.csv",
    }
    column_names = ["label", "text"]
    path_list = []
    datasets_list = []
    path, _ = cache_file(
        None,
        cache_dir=cache_dir,
        url=URL,
        md5sum=MD5,
        download_file_name="sogou_news_csv.tar.gz",
        proxies=proxies,
    )

    untar(path, cache_dir)
    if isinstance(split, str):
        path_list.append(os.path.join(cache_dir, "sogou_news_csv", path_dict[split]))
    else:
        for s in split:
            path_list.append(os.path.join(cache_dir, "sogou_news_csv", path_dict[s]))
    for path in path_list:
        datasets_list.append(
            GeneratorDataset(
                source=Sogounews(path), column_names=column_names, shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list
