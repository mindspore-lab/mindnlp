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
LCSTS load function
"""
# pylint: disable=C0103

import os
import json
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT

URL = {
    "train": "https://bj.bcebos.com/paddlenlp/datasets/LCSTS_new/train.json",
    "dev": "https://bj.bcebos.com/paddlenlp/datasets/LCSTS_new/dev.json",
}

MD5 = {
    "train": "4e06fd1cfd5e7f0380499df8cbe17237",
    "dev": "9c39d49d25d5296bdc537409208ddc85",
}


class Lcsts:
    """
    LCSTS dataset source
    """

    def __init__(self, path):
        self.path = path
        self._source, self._target = [], []
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding='utf8') as data:
            for line in data:
                line = line.strip()
                if not line:
                    continue
                json_data = json.loads(line)
                self._source.append(json_data["content"])
                self._target.append(json_data.get("summary", ''))

    def __getitem__(self, index):
        return self._source[index], self._target[index]

    def __len__(self):
        return len(self._source)

@load.register
def LCSTS(root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ('train', 'dev'), proxies=None):
    r"""
    Load the LCSTS dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'dev').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'dev')
        >>> dataset_train, dataset_dev = LCSTS(root, split)
        >>> train_iter = dataset_train.create_dict_iterator()
        >>> print(next(train_iter))
        {'source': Tensor(shape=[], dtype=String, value= '一辆小轿车，一名女司机，\
            竟造成9死24伤。日前，深圳市交警局对事故进行通报：从目前证据看，事故系司机超速行驶且操作不当导致。\
                目前24名伤员已有6名治愈出院，其余正接受治疗，预计事故赔偿费或超一千万元。'),
        'target': Tensor(shape=[], dtype=String, value= '深圳机场9死24伤续：司机全责赔偿或超千万')}

    """

    cache_dir = os.path.join(root, "datasets", "LCSTS")
    file_list = []
    datasets_list = []
    if isinstance(split, str):
        split = split.split()
    for key in split:
        path, _ = cache_file(
            None, url=URL[key], cache_dir=cache_dir, md5sum=MD5[key], proxies=proxies
        )
        file_list.append(path)

    for _, file in enumerate(file_list):
        dataset = GeneratorDataset(source=Lcsts(file), column_names=["source", "target"],
                                   shuffle=False)
        datasets_list.append(dataset)
    if len(file_list) == 1:
        return datasets_list[0]
    return datasets_list
