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
Hugging Face Ptb_text_only load function
"""
# pylint: disable=C0103
import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.utils import make_bucket
from mindnlp.transforms import BasicTokenizer, PadTransform, Truncate
from mindnlp.dataset.register import load_dataset, process
from mindnlp.dataset.process import common_process
from mindnlp.configs import DEFAULT_ROOT


class HFptb_text_only:
    """
    Hugging Face Ptb_text_only dataset source
    """

    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._sentence = []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._sentence.append(every_dict['sentence'])

    def __getitem__(self, index):
        return self._sentence[index]

    def __len__(self):
        return len(self._sentence)


@load_dataset.register
def HF_Ptb_text_only(
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = ("train", "validation", "test"),
        shuffle=True,
):
    r"""
    Load the huggingface Ptb_text_only dataset.

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
        >>> from mindnlp.dataset import HF_Ptb_text_only
        >>> split = ('train', 'test')
        >>> dataset_train, dataset_test = HF_Ptb_text_only(split=split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """
    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets", "hf_datasets", "Ptb_text_only")
    else:
        cache_dir = root
    column_names = ['sentence']

    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('ptb_text_only', split=mode_list, cache_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFptb_text_only(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list


@process.register
def HF_Ptb_text_only_Process(dataset, column="sentence", tokenizer=BasicTokenizer(), vocab=None,
                             batch_size=64, max_len=500, bucket_boundaries=None, drop_remainder=False):
    """
    the process of the Ptb_text_only dataset

    Args:
        dataset (GeneratorDataset): Ptb_text_only dataset.
        column (str): the column needed to be transpormed of the Ptb_text_only dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.
        batch_size (int): size of the batch.
        max_len (int): max length of the sentence.
        bucket_boundaries (list[int]): A list consisting of the upper boundaries of the buckets.
        drop_remainder (bool): If True, will drop the last batch for each bucket if it is not a full batch

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> from mindnlp.dataset import HF_Ptb_text_only, HF_Ptb_text_only_Process
        >>> dataset_train, dataset_test = HF_Ptb_text_only()
        >>> dataset_train = HF_Ptb_text_only_Process(dataset_train)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """

    if vocab is None:
        dataset, vocab = common_process(dataset, column, tokenizer, vocab)
    else:
        dataset = common_process(dataset, column, tokenizer, vocab)

    pad_value = vocab.tokens_to_ids("<pad>")
    trancate_op = Truncate(max_len)
    dataset = dataset.map([trancate_op], column)
    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(
                f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        if bucket_boundaries[-1] < max_len + 1:
            bucket_boundaries.append(max_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = make_bucket(dataset, column, pad_value,
                              bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_op = PadTransform(max_len, pad_value)
        dataset = dataset.map([pad_op], column)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset, vocab
