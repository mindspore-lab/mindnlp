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
Hugging Face IMDB load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
import mindspore as ms
from mindspore.dataset import GeneratorDataset, text, transforms
from mindnlp.transforms import Truncate
from mindnlp.dataset.register import load_dataset, process
from mindnlp.dataset.utils import make_bucket
from mindnlp.configs import DEFAULT_ROOT


class HFimdb:
    """
    Hugging Face IMDB dataset source
    """
    def __init__(self, dataset_list) -> None:
        self.dataset_list = dataset_list
        self._label, self._text = [], []
        self._load()

    def _load(self):
        for every_dict in self.dataset_list:
            self._label.append(every_dict['label'])
            self._text.append(every_dict['text'])

    def __getitem__(self, index):
        return self._text[index], self._label[index]

    def __len__(self):
        return len(self._label)


@load_dataset.register
def HF_IMDB(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "test"),
    shuffle=True,
):
    r"""
    Load the huggingface IMDB dataset.

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

    cache_dir = os.path.join(root, "datasets", "hf_datasets", "IMDB")
    column_names = ["text", "label"]
    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load('imdb', split=mode_list, data_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFimdb(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list


@process.register
def HF_IMDB_Process(dataset, tokenizer, vocab, batch_size=64, max_len=500, \
                    bucket_boundaries=None, drop_remainder=False):
    """
    the process of the IMDB dataset

    Args:
        dataset (GeneratorDataset): IMDB dataset.
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
        >>> imdb_train, imdb_test = load_dataset('imdb', shuffle=True)
        >>> embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)
        >>> tokenizer = BasicTokenizer(True)
        >>> imdb_train = process('hf_imdb', imdb_train, tokenizer=tokenizer, vocab=vocab, \
                        bucket_boundaries=[400, 500], max_len=600, drop_remainder=True)
    """

    pad_value = vocab.tokens_to_ids('<pad>')

    lookup_op = text.Lookup(vocab, unknown_token='<unk>')
    type_cast_op = transforms.TypeCast(ms.int32)
    dataset = dataset.map([tokenizer, lookup_op], 'text')
    dataset = dataset.map(type_cast_op, 'label')

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        trancate_op = Truncate(max_len)
        dataset = dataset.map([trancate_op], 'text')
        if bucket_boundaries[-1] < max_len + 1:
            bucket_boundaries.append(max_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = make_bucket(dataset, 'text', pad_value, \
                              bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_op = transforms.PadEnd([max_len], pad_value)
        dataset = dataset.map([pad_op], 'text')
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset
