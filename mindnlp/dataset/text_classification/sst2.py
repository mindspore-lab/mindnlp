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
SST2 dataset
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import unzip

URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

MD5 = "9f81648d4199384278b86e315dac217c"


class Sst2:
    """
    SST2 dataset source
    """

    def __init__(self, path) -> None:
        self.path: str = path
        self._label, self._text = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        lines.pop(0)
        lines.pop(len(lines) - 1)
        print(len(lines))
        if self.path.endswith("test.tsv"):
            for line in lines:
                l = line.split("\t")
                self._text.append(l[1])
        else:
            for line in lines:
                l = line.split("\t")
                self._text.append(l[0])
                self._label.append(l[1])

    def __getitem__(self, index):
        if self.path.endswith("test.tsv"):
            return self._text[index]
        return self._label[index], self._text[index]

    def __len__(self):
        return len(self._text)


@load.register
def SST2(
    root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "dev", "test"), proxies=None
):
    r"""
    Load the SST2 dataset
    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'dev', 'test').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Examples:
        >>> dataset_train,dataset_dev,dataset_test = SST2()
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= '0'), Tensor(shape=[], dtype=String, \
        value= 'hide new secretions from the parental units ')]
    """
    cache_dir = os.path.join(root, "datasets", "SST2")
    column_names = []
    path_list = []
    datasets_list = []
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, proxies=proxies)
    unzip(path, cache_dir)
    if isinstance(split, str):
        path_list.append(os.path.join(cache_dir, "SST-2", split + ".tsv"))
        if split == "test":
            column_names.append(["text"])
        else:
            column_names.append(["label", "text"])
    else:
        for s in split:
            path_list.append(os.path.join(cache_dir, "SST-2", s + ".tsv"))
            if split == "test":
                column_names.append(["text"])
            else:
                column_names.append(["label", "text"])
    for idx, path in enumerate(path_list):
        datasets_list.append(
            GeneratorDataset(
                source=Sst2(path), column_names=column_names[idx], shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list
