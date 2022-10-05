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
STSB dataset
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar

URL = "http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz"

MD5 = "4eb0065aba063ef77873d3a9c8088811"


class Stsb:
    """
    STSB dataset source
    """

    def __init__(self, path) -> None:
        self.path: str = path
        self._index,self._label,self._sentence1,self._sentence2 = [],[],[],[]
        self._load()

    def _load(self):
        with open(self.path, "r",encoding='utf-8')as f:
            dataset = f.read()
        lines = dataset.split("\n")
        lines.pop(len(lines)-1)
        for line in lines:
            l = line.split("\t")
            self._index.append(int(l[3]))
            self._label.append(float(l[4]))
            self._sentence1.append(l[5])
            self._sentence2.append(l[6])

    def __getitem__(self, index):
        return self._index[index], self._label[index], self._sentence1[
            index], self._sentence2[index]

    def __len__(self):
        return len(self._sentence1)

@load.register
def STSB(root: str = DEFAULT_ROOT,
         split: Union[Tuple[str], str] = ("train", "dev", "test"),
         proxies=None):
    r"""
    Load the STSB dataset
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
        >>> dataset_train,dataset_dev,dataset_test = STSB()
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Float64,
        value= 5), Tensor(shape=[], dtype=String, value= 'A plane is taking off.'),
        Tensor(shape=[], dtype=String, value= 'An air plane is taking off.')]

    """
    cache_dir = os.path.join(root, "datasets", "STSB")
    column_names = ["index", "label", "sentence1", "sentence2"]
    path_list = []
    datasets_list = []
    path, _ = cache_file(None,
                         url=URL,
                         cache_dir=cache_dir,
                         md5sum=MD5,
                         proxies=proxies)
    untar(path, cache_dir)
    if isinstance(split, str):
        path_list.append(
            os.path.join(cache_dir, "stsbenchmark", f"sts-{split}.csv"))
    else:
        for s in split:
            path_list.append(
                os.path.join(cache_dir, "stsbenchmark", f"sts-{s}.csv"))
    for path in path_list:
        datasets_list.append(
            GeneratorDataset(source=Stsb(path),
                             column_names=column_names,
                             shuffle=False))
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list
