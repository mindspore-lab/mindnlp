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
PennTreebank load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import PennTreebankDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT

URL = {
    "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "valid": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
    "test": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
}

MD5 = {
    "train": "f26c4b92c5fdc7b3f8c7cdcb991d8420",
    "valid": "aa0affc06ff7c36e977d7cd49e3839bf",
    "test": "8b80168b89c18661a38ef683c0dc3721",
}


@load.register
def PennTreebank(root: str = DEFAULT_ROOT,
                 split: Union[Tuple[str], str] = ('train', 'valid', 'test'), proxies=None):
    r"""
    Load the PennTreebank dataset

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
        >>> dataset_train, dataset_valid, dataset_test = PennTreebank(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= ' aer banknote berlitz calloway centrust \
            cluett fromstein gitano guterman hydro-quebec ipo kia memotec mlx nahb punts \
                rake regatta rubens sim snack-food ssangyong swapo wachter ')]

    """
    cache_dir = os.path.join(root, "datasets", "PennTreebank")
    datasets_list = []

    for key, value in URL.items():
        cache_file(None, cache_dir=cache_dir, url=value, md5sum=MD5[key], proxies=proxies)
    if isinstance(split, str):
        split = split.split()
    for s in split:
        dataset = PennTreebankDataset(
            dataset_dir=cache_dir, usage=s, shuffle=False)
        datasets_list.append(dataset)
    if len(datasets_list) == 1:
        return datasets_list[0]
    return datasets_list
