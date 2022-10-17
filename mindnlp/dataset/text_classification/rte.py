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
RTE dataset
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import unzip

URL = "https://dl.fbaipublicfiles.com/glue/data/RTE.zip"

MD5 = "bef554d0cafd4ab6743488101c638539"


class Rte:
    """
    RTE dataset source
    """

    label_map = {"entailment": 0, "not_entailment": 1}

    def __init__(self, path) -> None:
        self.path: str = path
        self._label,self._sentence1,self._sentence2 = [],[],[]
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        lines.pop(0)
        lines.pop(len(lines)-1)
        if self.path.endswith("test.tsv"):
            for line in lines:
                l = line.split('\t')
                self._sentence1.append(l[1])
                self._sentence2.append(l[2])
        else:
            for line in lines:
                l = line.split('\t')
                self._sentence1.append(l[1])
                self._sentence2.append(l[2])
                self._label.append(self.label_map[l[3]])


    def __getitem__(self,index):
        if self.path.endswith("test.tsv"):
            return self._sentence1[index],self._sentence2[index]
        return self._label[index],self._sentence1[index],self._sentence2[index]

    def __len__(self):
        return len(self._sentence1)


@load.register
def RTE(
    root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "dev", "test"), proxies=None
):
    r"""
    Load the WNLI dataset
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
        >>> dataset_train,dataset_dev,dataset_test = RTE()
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    cache_dir = os.path.join(root, "datasets", "CoLA")
    path_dict = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }
    column_names_dict = {
        "train": ["label","sentence1","sentece2"],
        "dev": ["label","sentence1","sentece2"],
        "test": ["sentence1","sentece2"],
    }
    column_names = []
    path_list = []
    datasets_list = []
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, proxies=proxies)
    unzip(path, cache_dir)
    if isinstance(split, str):
        path_list.append(
            os.path.join(cache_dir, "RTE", path_dict[split])
        )
        column_names.append(column_names_dict[split])
    else:
        for s in split:
            path_list.append(
                os.path.join(cache_dir, "RTE", path_dict[s])
            )
            column_names.append(column_names_dict[s])
    for idx, path in enumerate(path_list):
        datasets_list.append(
            GeneratorDataset(
                source=Rte(path), column_names=column_names[idx], shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list
