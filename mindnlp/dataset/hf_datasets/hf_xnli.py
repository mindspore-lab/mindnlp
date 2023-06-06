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
Hugging Face Xnli load function
"""
# pylint: disable=C0103
import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.register import load_dataset
from mindnlp.configs import DEFAULT_ROOT


class HFxnli:
    """
    Hugging Face Xnli dataset source
    """

    def __init__(self, dataset_list, name) -> None:
        self.dataset_list = dataset_list
        self._premise, self._hypothesis, self._label = [], [], []
        self._load(name)

    def _load(self, name):
        for every_dict in self.dataset_list:
            self._label.append(every_dict['label'])
            if name != "all_languages":
                self._premise.append(every_dict['premise'])
                self._hypothesis.append(every_dict['hypothesis'])
            else:
                languages = every_dict['hypothesis']['languages']
                for i, lg in enumerate(languages):
                    self._premise.append(every_dict['premise'][lg])
                    self._hypothesis.append(every_dict['hypothesis']['translation'][i])

    def __getitem__(self, index):
        return self._premise[index], self._hypothesis[index], self._label[index]

    def __len__(self):
        return len(self._premise)


@load_dataset.register
def HF_Xnli(
        name: str,
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = ("train", "validation", "test"),
        shuffle=True,
):
    r"""
    Load the huggingface Xnli dataset.

    Args:
        name (str):Task name
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'validation', 'test').
        shuffle (bool): Whether to shuffle the dataset.
            Default:True.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> from mindnlp.dataset import HF_Xnli
        >>> split = ("train", "validation", "test")
        >>> dataset_train, _, _ = HF_Ptb_text_only(name='zh', split=split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """
    cache_dir = os.path.join(root, "datasets", "hf_datasets", "Xnli")
    column_names = ['premise', 'hypothesis', 'label']

    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('xnli', name, split=mode_list, cache_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFxnli(every_ds, name),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list
