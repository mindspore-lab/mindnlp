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
Hugging Face mt_eng_vietnamese load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.register import load_dataset
from mindnlp.configs import DEFAULT_ROOT


class HF_mt_eng_vietnamese:
    """
    Hugging Face mt_eng_vietnamese dataset source
    """
    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._translation = []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._translation.append(every_dict['translation'])

    def __getitem__(self, index):
        return self._translation[index]

    def __len__(self):
        return len(self._translation)


@load_dataset.register
def hf_mt_eng_vietnamese(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "validation", "test"),
    shuffle=True,
):
    r"""
    Load the huggingface mt_eng_vietnamese dataset.

    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'test').
        shuffle (bool): Whether to shuffle the dataset.
            Default:True.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'test')
        >>> dataset_train,dataset_test = HF_IMDB(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets", "hf_datasets", "mt_eng_vietnamese")
    else:
        cache_dir = root
    column_names = ["translation"]
    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('mt_eng_vietnamese', 'iwslt2015-en-vi',split=mode_list, data_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HF_mt_eng_vietnamese(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list
