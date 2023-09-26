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
from mindnlp.dataset.register import load_dataset, process
from mindnlp.transforms import BasicTokenizer
from mindnlp.dataset.process import common_process
from mindnlp.configs import DEFAULT_ROOT


class HFmt_eng_vietnamese:
    """
    Hugging Face mt_eng_vietnamese dataset source
    """
    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._en, self._vi = [], []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._en.append(every_dict['translation']['en'])
            self._vi.append(every_dict['translation']['vi'])

    def __getitem__(self, index):
        return self._en[index], self._vi[index]

    def __len__(self):
        return len(self._en)


@load_dataset.register
def Hf_mt_eng_vietnamese(
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
        >>> dataset_train,dataset_test = Hf_mt_eng_vietnamese(
            root=self.root, split=("train", "validation", "test")
        )
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets", "hf_datasets", "mt_eng_vietnamese")
    else:
        cache_dir = root
    column_names = ["en", "vi"]
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
            source=HFmt_eng_vietnamese(every_ds),
            column_names=column_names, shuffle=shuffle)
        )

    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list



@process.register
def Hf_mt_eng_vietnamese_Process(dataset, column="en", tokenizer=BasicTokenizer(),vocab=None):
    """
    the process of the mt_eng_vietnamese dataset

    Args:
        dataset (GeneratorDataset): mt_eng_vietnamese dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.
        batch_size (int): size of the batch.
        max_len (int): max length of the sentence.
        bucket_boundaries (list[int]): A list consisting of the upper boundaries of the buckets.
        drop_remainder (bool): If True, will drop the last batch for each bucket if it is not a full batch

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.
        - **Vocab** (Vocab) - vocab created from dataset

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> mt_eng_vietnamese_train, mt_eng_vietnamese_test = load('mt_eng_vietnamese', shuffle=True)
        >>> embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)
        >>> tokenizer = BasicTokenizer(True)
        >>> mt_eng_vietnamese_train = process('Hf_mt_eng_vietnamese',
                                       dataset=train_dataset,
                                       column="en",
                                       tokenizer=BasicTokenizer(),
                                       vocab=None
                                       )
    """
    if vocab is None:
        dataset, vocab = common_process(dataset, column, tokenizer, vocab)
    else:
        dataset = common_process(dataset, column, tokenizer, vocab)

    return dataset,vocab
