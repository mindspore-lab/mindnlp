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
Hugging Face Funsd load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.register import load_dataset
from mindnlp.configs import DEFAULT_ROOT


class HFfunsd:
    """
    Hugging Face Funsd dataset source
    """
    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._id = []
        self._words = []
        self._bboxes = []
        self._ner_tags = []
        self._image_path = []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._id.append(every_dict['id'])
            self._words.append(every_dict['words'])
            self._bboxes.append(every_dict['bboxes'])
            self._ner_tags.append(every_dict['ner_tags'])
            self._image_path.append(every_dict['image_path'])

    def __getitem__(self, index):
        return self._id[index], self._words[index], self._bboxes[index], self._ner_tags[index], self._image_path[index]

    def __len__(self):
        return len(self._id)


@load_dataset.register
def HF_FUNSD(
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = ('train', 'test')
):
    r"""
    Load the huggingface Funsd dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'test').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'test')
        >>> dataset_train, dataset_test  = HF_FUNSD(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets", "funsd")
    else:
        cache_dir = root
    datasets_list = []
    mode_list = []
    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('nielsr/funsd', split=mode_list, cache_dir=cache_dir)
    for _, file in enumerate(ds_list):
        dataset = GeneratorDataset(source=HFfunsd(file),
                                   column_names=[
                                       "id", "words", "bboxes", "ner_tags", "image_path"],
                                   shuffle=False)
        datasets_list.append(dataset)
    if len(ds_list) == 1:
        return datasets_list[0]
    return datasets_list
