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
Hugging Face SQuAD2 load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.text_classification.imdb import IMDB_Process
from mindnlp.dataset.register import load_dataset, process
from mindnlp.configs import DEFAULT_ROOT


class HFsquad2:
    """
    Hugging Face IMDB dataset source
    """
    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._context, self._question = [], []
        self._anwsers, self._answers_start = [], []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._context.append(every_dict['context'])
            self._question.append(every_dict['question'])
            self._anwsers.append(every_dict['answers']['text'])
            self._answers_start.append(every_dict['answers']['answer_start'])

    def __getitem__(self, index):
        return self._context[index], self._question[index],\
            self._anwsers[index], self._answers_start[index]

    def __len__(self):
        return len(self._question)


@load_dataset.register
def HF_SQuAD2(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "validation"),
    shuffle=True,
):
    r"""
    Load the SQuAD2 dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train','validation').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'validation')
        >>> dataset_train, dataset_validation = HF_SQuAD2(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= 'Beyoncé Giselle Knowles-Carter...),
        Tensor(shape=[], dtype=String, value= 'When did Beyonce start becoming popular?'),
        Tensor(shape=[1], dtype=String, value= ['in the late 1990s']),
        Tensor(shape=[1], dtype=Int32, value= [269])]

    """

    cache_dir = os.path.join(root, "datasets", "hf_datasets", "SQuAD2")
    column_names = ["context", "question", "answers", "answers_start"]
    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('squad_v2', split=mode_list, data_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFsquad2(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list

