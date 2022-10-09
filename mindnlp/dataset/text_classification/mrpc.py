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
MRPC load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT

URL = {
    "train": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
    "test": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt",
}

MD5 = {
    "train": "793daf7b6224281e75fe61c1f80afe35",
    "test": "e437fdddb92535b820fe8852e2df8a49",
}

class Mrpc:
    """
    MRPC dataset source
    """

    def __init__(self, path):
        self.path = path
        self._label, self._sentence1, self._sentence2 = [], [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        lines.pop(0)
        lines.pop(len(lines) - 1)
        for line in lines:
            l = line.split("\t")
            self._label.append(int(l[0]))
            self._sentence1.append(l[3])
            self._sentence2.append(l[4])

    def __getitem__(self, index):
        return self._label[index], self._sentence1[index], self._sentence2[index]

    def __len__(self):
        return len(self._label)

@load.register
def MRPC(root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "test"), proxies=None):
    r"""
    Load the AG_NEWS dataset
    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'test').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        >>> dataset_train,dataset_test = MRPC()
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    cache_dir = os.path.join(root, "datasets", "MRPC")
    column_names = ["label", "sentence1", "sentence2"]
    datasets_list = []
    path_list = []
    if isinstance(split,str):
        path, _ = cache_file(None, url=URL[split], cache_dir=cache_dir, md5sum=MD5[split], proxies=proxies)
        path_list.append(path)
    else:
        for s in split:
            path, _ = cache_file(None, url=URL[s], cache_dir=cache_dir, md5sum=MD5[s], proxies=proxies)
            path_list.append(path)
    for path in path_list:
        datasets_list.append(GeneratorDataset(source=Mrpc(path), column_names=column_names, shuffle=False))
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list
