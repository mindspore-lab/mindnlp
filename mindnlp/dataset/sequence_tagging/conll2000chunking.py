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
CoNLL2000Chunking load function
"""
# pylint: disable=C0103

import os
import re
from typing import Union, Tuple
import mindspore
from mindspore.dataset import GeneratorDataset, text, transforms
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load, process
from mindnlp.dataset.utils import make_bucket_2cloums
from mindnlp.dataset.transforms.seq_process import TruncateSequence
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import ungz

URL = {
    "train": "https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz",
    "test": "https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz",
}

MD5 = {
    "train": "6969c2903a1f19a83569db643e43dcc8",
    "test": "a916e1c2d83eb3004b38fc6fcd628939",
}


class Conll2000chunking:
    """
    CoNLL2000Chunking dataset source
    """

    def __init__(self, path):
        self.path = path
        self._words, self._tag, self._chunk_tag = [], [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        tmp_words = []
        tmp_tag = []
        tmp_chunk_tag = []
        for line in lines:
            if line == "":
                if tmp_words:
                    self._words.append(tmp_words)
                    self._tag.append(tmp_tag)
                    self._chunk_tag.append(tmp_chunk_tag)
                    tmp_words = []
                    tmp_tag = []
                    tmp_chunk_tag = []
                else:
                    break
            else:
                l = line.split(" ")
                tmp_words.append(l[0])
                tmp_tag.append(l[1])
                tmp_chunk_tag.append(l[2])

    def __getitem__(self, index):
        return self._words[index], self._tag[index], self._chunk_tag[index]

    def __len__(self):
        return len(self._words)


@load.register
def CoNLL2000Chunking(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "test"),
    proxies=None,
):
    r"""
    Load the CoNLL2000Chunking dataset
    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'test').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'test')
        >>> dataset_train,dataset_test = CoNLL2000Chunking(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    cache_dir = os.path.join(root, "datasets", "CoNLL2000Chunking")
    column_names = ["words", "tag", "chunk_tag"]
    datasets_list = []
    path_list = []
    if isinstance(split, str):
        path, _ = cache_file(
            None,
            url=URL[split],
            cache_dir=cache_dir,
            md5sum=MD5[split],
            proxies=proxies,
        )
        path_list.append(ungz(path))
    else:
        for s in split:
            path, _ = cache_file(
                None, url=URL[s], cache_dir=cache_dir, md5sum=MD5[s], proxies=proxies
            )
            path_list.append(ungz(path))
    for path in path_list:
        datasets_list.append(
            GeneratorDataset(
                source=Conll2000chunking(path), column_names=column_names, shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def CoNLL2000Chunking_Process(dataset, vocab, batch_size=64, max_len=500, \
                 bucket_boundaries=None, drop_remainder=False):
    """
    the process of the CoNLL2000Chunking dataset

    Args:
        dataset (GeneratorDataset): CoNLL2000Chunking dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> dataset_train,dataset_test = CoNLL2000Chunking()
        >>> vocab = text.Vocab.from_dataset(dataset_train,columns=["words"],freq_range=None,
                                    top_k=None,special_tokens=["<pad>","<unk>"],special_first=True)
        >>> dataset_train = CoNLL2000Chunking_Process(dataset=dataset_train, vocab=vocab,
                                          batch_size=32, max_len=80)
    """
    columns_to_project = ["words", "chunk_tag"]
    dataset = dataset.project(columns=columns_to_project)
    input_columns = ["words", "chunk_tag"]
    output_columns = ["text", "label"]
    dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)

    class TmpDataset:
        """ a Dataset for seq_length column """
        def __init__(self, dataset):
            self._dataset = dataset
            self._seq_length = []
            self._load()

        def _load(self):
            for data in self._dataset.create_dict_iterator():
                self._seq_length.append(len(data["text"]))

        def __getitem__(self, index):
            return self._seq_length[index]

        def __len__(self):
            return len(self._seq_length)

    dataset_tmp = GeneratorDataset(TmpDataset(dataset), ["seq_length"],shuffle=False)
    dataset = dataset.zip(dataset_tmp)
    columns_order = ["text", "seq_length", "label"]
    dataset = dataset.project(columns=columns_order)

    pad_value_text = vocab.tokens_to_ids('<pad>')
    pad_value_label = 0
    lookup_op = text.Lookup(vocab, unknown_token='<unk>')
    type_cast_op = transforms.TypeCast(mindspore.int64)

    def tag_idx(tags):
        """ tag_idx """
        tag_idx_list = []
        regex_dic = {"O":0,"B-ADJP":1,"I-ADJP":2,"B-ADVP":3,"I-ADVP":4,"B-CONJP":5,
                     "I-CONJP":6,"B-INTJ":7,"I-INTJ":8,"B-LST":9,"I-LST":10,"B-NP":11,
                     "I-NP":12,"B-PP":13,"I-PP":14,"B-PRT":15,"I-PRT":16,"B-SBAR":17,
                     "I-SBAR":18,"B-UCP":19,"I-UCP":20,"B-VP":21,"I-VP":22}
        for tag in tags:
            for key, value in regex_dic.items():
                if re.match(key, tag):
                    tag_idx_list.append(value)
        return tag_idx_list

    dataset = dataset.map(operations=[tag_idx], input_columns=["label"])
    dataset = dataset.map(operations=[lookup_op], input_columns=["text"])
    dataset = dataset.map(operations=[type_cast_op])

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        trancate_op = TruncateSequence(max_len)
        dataset = dataset.map([trancate_op], 'text')
        dataset = dataset.map([trancate_op], 'label')
        dataset.get_dataset_size()
        if bucket_boundaries[-1] < max_len + 1:
            bucket_boundaries.append(max_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset =  make_bucket_2cloums(dataset, ['text','label'], pad_value_text, pad_value_label, \
                              bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_op_text = transforms.PadEnd([max_len], pad_value_text)
        pad_op_label = transforms.PadEnd([max_len], pad_value_label)
        dataset = dataset.map([pad_op_text], 'text')
        dataset = dataset.map([pad_op_label], 'label')
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset
