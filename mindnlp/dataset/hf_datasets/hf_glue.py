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
Hugging Face GLUE load function
"""
# pylint: disable=C0103
import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset, text
from mindnlp.transforms import BasicTokenizer
from mindnlp.dataset.process import common_process
from mindnlp.dataset.register import load_dataset, process
from mindnlp.configs import DEFAULT_ROOT


class HFglue:
    """
    Hugging Face GLUE dataset source
    """

    def __init__(self, dataset_list, name) -> None:
        self.dataset_list = dataset_list
        if name in ('cola', 'sst2'):
            self._label, self._idx, self._sentence = [], [], []
        elif name in ('mrpc', 'stsb', 'rte', 'wnli'):
            self._label, self._idx, self._sentence1, self._sentence2 = [], [], [], []
        elif name == "qqp":
            self._label, self._idx, self._question1, self._question2 = [], [], [], []
        elif (len(name) >= 4 and name[0:4] == "mnli") or name == "ax":
            self._label, self._idx, self._premise, self._hypothesis = [], [], [], []
        elif name == "qnli":
            self._label, self._idx, self._question, self._sentence = [], [], [], []
        self._label, self._text = [], []
        self._load(name)

    def _load(self, name):
        for every_dict in self.dataset_list:
            self._label.append(every_dict['label'])
            self._text.append(every_dict['idx'])
            if name in ('cola', 'sst2'):
                self._sentence.append(every_dict['sentence'])
            elif name in ('mrpc', 'stsb', 'rte', 'wnli'):
                self._sentence1.append(every_dict['sentence1'])
                self._sentence2.append(every_dict['sentence2'])
            elif name == "qqp":
                self._question1.append(every_dict['question1'])
                self._question2.append(every_dict['question2'])
            elif (len(name) >= 4 and name[0:4] == "mnli") or name == "ax":
                self._premise.append(every_dict['premise'])
                self._hypothesis.append(every_dict['hypothesis'])
            elif name == "qnli":
                self._sentence.append(every_dict['sentence'])
                self._question.append(every_dict['question'])

    def __getitem__(self, index):
        return self._text[index], self._label[index]

    def __len__(self):
        return len(self._label)


@load_dataset.register
def HF_GLUE(
        name: str,
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = ("train", "test"),
        shuffle=True,
):
    r"""
    Load the huggingface GLUE dataset.

    Args:
        name (str):Task name
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
        >>> dataset_train,dataset_test = HF_GLUE(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets", "hf_datasets", "GLUE")
    else:
        cache_dir = root
    if name in ('cola', 'sst2'):
        column_names = ["label", "sentence"]
    elif name in ('mrpc', 'stsb', 'rte', 'wnli'):
        column_names = {
            "train": ["label", "sentence1", "sentence2"],
            "validation": ["label", "sentence1", "sentence2"],
            "test": ["sentence1", "sentence2"],
        }
    elif name == "qqp":
        column_names = ['question1', 'question2', 'label']
    elif (len(name) >= 4 and name[0:4] == "mnli") or name == "ax":
        column_names = ['premise', 'hypothesis', 'label']
    elif name == "qnli":
        column_names = {
            "train": ["label", "question", "sentence"],
            "validation": ["label", "question", "sentence"],
            "test": ["question", "sentence"],
        }

    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('glue', name, split=mode_list, cache_dir=cache_dir)
    if name in ('mrpc', 'stsb', 'rte', 'wnli', 'qnli'):
        flag = 0
        for every_ds in ds_list:
            datasets_list.append(GeneratorDataset(
                source=HFglue(every_ds, name),
                column_names=column_names[split[flag]], shuffle=shuffle)
            )
            flag += 1
    else:
        for every_ds in ds_list:
            datasets_list.append(GeneratorDataset(
                source=HFglue(every_ds, name),
                column_names=column_names, shuffle=shuffle)
            )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def HF_GLUE_Process(name, dataset, column=None, tokenizer=BasicTokenizer(True), vocab=None):
    """
    the process of the GLUE dataset
    """
    if name in ('cola', 'sst2'):
        return common_process(dataset, 'sentence', tokenizer, vocab)
    if column is None:
        if name in ('mrpc', 'stsb', 'rte', 'wnli'):
            column = ['sentence1', 'sentence2']
        elif name == "qqp":
            column = ['question1', 'question2']
        elif (len(name) >= 4 and name[0:4] == "mnli") or name == "ax":
            column = ['premise', 'hypothesis']
        elif name == "qnli":
            column = ['question', 'sentence']
    if vocab is None:
        for col in column:
            dataset = dataset.map(tokenizer, input_columns=col)
        column = list(column)
        vocab = text.Vocab.from_dataset(dataset, columns=column, special_tokens=["<pad>", "<unk>"])
        for col in column:
            dataset = dataset.map(text.Lookup(vocab, unknown_token='<unk>'), input_columns=col)
        return dataset, vocab
    for col in column:
        dataset = dataset.map(tokenizer, input_columns=col)
    for col in column:
        dataset = dataset.map(text.Lookup(vocab, unknown_token='<unk>'), input_columns=col)
    return dataset, vocab
