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
UDPOS load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import unzip

URL = "https://bitbucket.org/sivareddyg/public/downloads/en-ud-v2.zip"

MD5 = "bdcac7c52d934656bae1699541424545"


class Udpos:
    """
    UDPOS dataset source
    """

    def __init__(self, path) -> None:
        self.path = path
        self._words, self._tag, self._chunk_tag = [], [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        tmp_words = []
        tmp_tag = []
        tmp_chunk_tag = []
        for line in lines:
            if line == "":
                if tmp_words:
                    self._words.append(tmp_words)
                    self._tag.append(tmp_tag)
                    self._chunk_tag.append(tmp_chunk_tag)
                    tmp_words = []
                    tmp_tag = []
                    tmp_chunk_tag = []
                else:
                    break
            else:
                l = line.split("\t")
                tmp_words.append(l[0])
                tmp_tag.append(l[1])
                tmp_chunk_tag.append(l[2])

    def __getitem__(self, index):
        return self._words[index], self._tag[index], self._chunk_tag[index]

    def __len__(self):
        return len(self._words)


@load.register
def UDPOS(
    root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "dev", "test"), proxies=None
):
    r"""
    Load the UDPOS dataset

    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'dev', 'test').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'dev', 'test')
        >>> dataset_train,dataset_dev,dataset_test = UDPOS(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    cache_dir = os.path.join(root, "datasets", "UDPOS")
    path_dict = {
        "train": "en-ud-tag.v2.train.txt",
        "dev": "en-ud-tag.v2.dev.txt",
        "test": "en-ud-tag.v2.test.txt",
    }
    column_names = ["words", "tag", "chunk_tag"]
    path_list = []
    datasets_list = []
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, proxies=proxies)
    unzip(path, cache_dir)
    if isinstance(split, str):
        path_list.append(
            os.path.join(cache_dir, "en-ud-v2", path_dict[split])
        )
    else:
        for s in split:
            path_list.append(
                os.path.join(cache_dir, "en-ud-v2", path_dict[s])
            )
    for path in path_list:
        datasets_list.append(
            GeneratorDataset(
                source=Udpos(path), column_names=column_names, shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list
