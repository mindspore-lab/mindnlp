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
Hugging Face Msra_ner load function
"""
# pylint: disable=C0103
import os
from typing import Union, Tuple
import numpy as np
from datasets import load_dataset as hf_load
import mindspore as ms
from mindspore.dataset import GeneratorDataset, transforms
from mindnlp.dataset.utils import make_bucket_2cloums
from mindnlp.transforms import PadTransform, Truncate
from mindnlp.dataset.register import load_dataset, process
from mindnlp.configs import DEFAULT_ROOT


class HFmsra_ner:
    """
    Hugging Face Msra_ner dataset source
    """

    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._ner_tags, self._id, self._tokens = [], [], []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._ner_tags.append(every_dict['ner_tags'])
            self._id.append(every_dict['id'])
            self._tokens.append(every_dict['tokens'])

    def __getitem__(self, index):
        return self._tokens[index], self._ner_tags[index]

    def __len__(self):
        return len(self._ner_tags)


@load_dataset.register
def HF_Msra_ner(
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = ("train", "test"),
        shuffle=True,
):
    r"""
    Load the huggingface Msra_ner dataset.

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
        >>> from mindnlp.dataset import HF_Msra_ner
        >>> split = ('train', 'test')
        >>> dataset_train,dataset_test = HF_Msra_ner(split=split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """
    cache_dir = os.path.join(root, "datasets", "hf_datasets", "Msra_ner")
    column_names = ['tokens', 'ner_tags']

    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('msra_ner', split=mode_list, cache_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFmsra_ner(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list


@process.register
def HF_Msra_ner_Process(dataset, tokenizer, batch_size=64, max_len=500,
                        bucket_boundaries=None, drop_remainder=False):
    """
    the process of the Msra_ner dataset

    Args:
        dataset (GeneratorDataset): Msra_ner dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        batch_size (int): size of the batch.
        max_len (int): max length of the sentence.
        bucket_boundaries (list[int]): A list consisting of the upper boundaries of the buckets.
        drop_remainder (bool): If True, will drop the last batch for each bucket if it is not a full batch

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.
        input_columns = ["tokens", "ner_tags"], input_columns = ["tokens", "seq_length", "ner_tags"].

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> from mindnlp.transforms import BertTokenizer
        >>> from mindnlp.dataset import HF_Msra_ner, HF_Msra_ner_Process
        >>> dataset_train,dataset_test = HF_Msra_ner()
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        >>> dataset_train = HF_Msra_ner_Process(dataset_train, tokenizer=tokenizer, \
                            batch_size=64, max_len=512)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """
    pad_value_tokens = tokenizer.pad_token_id
    pad_value_label = 0

    trancate_op = Truncate(max_len-2)
    type_cast_op = transforms.TypeCast(ms.int64)

    def add_cls_sep_tokens(x):
        cls = tokenizer.cls_token_id
        sep = tokenizer.sep_token_id
        x = np.insert(x, 0, cls)
        x = np.append(x, sep)
        return x

    def add_cls_sep_label(x):
        cls = 0
        sep = 0
        x = np.insert(x, 0, cls)
        x = np.append(x, sep)
        return x
    dataset = dataset.map([tokenizer.convert_tokens_to_ids, trancate_op, add_cls_sep_tokens], 'tokens')
    dataset = dataset.map(lambda x: (x, len(x)), input_columns='tokens', output_columns=['tokens', 'seq_length'])
    dataset = dataset.map([type_cast_op], 'seq_length')
    dataset = dataset.map([trancate_op, add_cls_sep_label, type_cast_op], 'ner_tags')

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(
                f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        if bucket_boundaries[-1] < max_len + 1:
            bucket_boundaries.append(max_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = make_bucket_2cloums(dataset, ['tokens', 'ner_tags'], pad_value_tokens, pad_value_label,
                                      bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_tokens_op = PadTransform(max_len, pad_value_tokens)
        pad_label_op = PadTransform(max_len, pad_value_label)
        dataset = dataset.map([pad_tokens_op], 'tokens')
        dataset = dataset.map([pad_label_op], 'ner_tags')
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset
