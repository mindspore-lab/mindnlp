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
Hugging Face dureader_robust load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from datasets import load_dataset as hf_load
from mindspore.dataset import GeneratorDataset, transforms

from mindnlp.transforms import Lookup, Truncate
from mindnlp.dataset.register import load_dataset, process
from mindnlp.dataset.utils import make_bucket
from mindnlp.configs import DEFAULT_ROOT


class HFdureader_robust:
    """
    Hugging Face dureader_robust dataset source
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
            if len(every_dict['answers']['answer_start']) == 0:
                self._anwsers.append('no answer')
                self._answers_start.append(-1)
            else:
                self._anwsers.append(every_dict['answers']['text'][0])
                self._answers_start.append(every_dict['answers']['answer_start'][0])

    def __getitem__(self, index):
        return self._context[index], self._question[index], \
            self._anwsers[index], self._answers_start[index]

    def __len__(self):
        return len(self._question)


@load_dataset.register
def HF_dureader_robust(
        root: str = DEFAULT_ROOT,
        split: Union[Tuple[str], str] = ("train", "validation", "test"),
        shuffle=True,
):
    r"""
    Load the dureader_robust dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train','validation', 'test').

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
        >>> dataset_train, dataset_validation = HF_dureader_robust(root, split, shuffle=False)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
    """

    cache_dir = os.path.join(root, "datasets", "hf_datasets", "dureader_robust")
    column_names = ["context", "question", "answers", "answers_start"]
    datasets_list = []
    mode_list = []

    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)

    ds_list = hf_load("PaddlePaddle/dureader_robust", split=mode_list, data_dir=cache_dir)
    for every_ds in ds_list:
        datasets_list.append(GeneratorDataset(
            source=HFdureader_robust(every_ds),
            column_names=column_names, shuffle=shuffle)
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list


@process.register
def HF_dureader_robust_Process(dataset, tokenizer, vocab, batch_size=64, max_context_len=1000, max_qa_len=30,
                      bucket_boundaries=None, drop_remainder=False):
    """
    the process of the SQuAD dataset

    Args:
        dataset (GeneratorDataset): dureader_robust dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.
        batch_size (int): size of the batch.
        max_context_len (int): max length of the context.
        max_qa_len (int): max length of the quention and answer.
        bucket_boundaries (list[int]): A list consisting of the upper boundaries of the buckets.
        drop_remainder (bool): If True, will drop the last batch for each bucket if it is not a full batch

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> dataset_train = HF_dureader_robust(split = ('train',))
        >>> tokenizer = mindnlp.transforms.BasicTokenizer(True)
        >>> dataset_train = dataset_train.map([tokenizer], 'context')
        >>> vocab = mindspore.dataset.text.Vocab.from_dataset(dataset_train, 'context', special_tokens=
        >>>   ['<pad>', '<unk>'], special_first= True)
        >>> squad_train = HF_dureader_robust_Process(dataset_train, tokenizer=tokenizer, vocab=vocab, \
                        bucket_boundaries=[400, 500], max_len=1000, drop_remainder=True)
    """

    pad_value = vocab.tokens_to_ids('<pad>')
    lookup_op = Lookup(vocab, unk_token='<unk>')
    dataset = dataset.map([tokenizer, lookup_op], 'context')
    dataset = dataset.map([tokenizer, lookup_op], 'question')
    dataset = dataset.map([tokenizer, lookup_op], 'answers')

    pad_qa_op = transforms.PadEnd([max_qa_len], pad_value)
    dataset = dataset.map([pad_qa_op], 'question')
    dataset = dataset.map([pad_qa_op], 'answers')

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        trancate_context_op = Truncate(max_context_len)
        dataset = dataset.map([trancate_context_op], 'context')
        trancate_qa_op = Truncate(max_qa_len)
        dataset = dataset.map([trancate_qa_op], 'question')
        dataset = dataset.map([trancate_qa_op], 'answers')

        if bucket_boundaries[-1] < max_context_len + 1:
            bucket_boundaries.append(max_context_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = make_bucket(dataset, 'context', pad_value,
                              bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_context_op = transforms.PadEnd([max_context_len], pad_value)
        dataset = dataset.map([pad_context_op], 'context')

        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
