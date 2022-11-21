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
AG_NEWS load function
"""
# pylint: disable=C0103
import html
import os
import re
import csv
from typing import Union, Tuple
import mindspore
from mindspore.dataset import GeneratorDataset, text, transforms
from mindnlp.dataset.transforms import TruncateSequence
from mindnlp.dataset.transforms import BasicTokenizer
from mindnlp.utils.download import cache_file
from mindnlp.dataset.utils import make_bucket
from mindnlp.dataset.register import load, process
from mindnlp.configs import DEFAULT_ROOT


URL = {
    "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
}

MD5 = {
    "train": "b1a00f826fdfbd249f79597b59e1dc12",
    "test": "d52ea96a97a2d943681189a97654912d",
}


class Agnews:
    """
    AG_NEWS dataset source
    """

    def __init__(self, path):
        self.path = path
        self._label, self._text = [], []
        self.end_string = ['.', '?', '!']
        self._load()

    def _load(self):
        csvfile = open(self.path, "r", encoding="utf-8")
        dict_reader = csv.reader(csvfile)
        for row in dict_reader:
            label = int(row[0]) - 1
            self._label.append(label)
            src_text1 = row[1]
            src_text2 = row[2]
            if src_text2:
                src_text2 = src_text2.strip()
            if src_text1 and src_text1[-1] not in self.end_string:
                src_text1 = src_text1 + '.'
            self._text.append(f"{src_text1} {src_text2}")

    def __getitem__(self, index):
        return self._label[index], self._text[index]

    def __len__(self):
        return len(self._text)


@load.register
def AG_NEWS(root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "test"), proxies=None, shuffle=False):
    r"""
    Load the AG_NEWS dataset

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
        >>> dataset_train,dataset_test = AG_NEWS(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= '3'), Tensor(shape=[], dtype=String,\
             value= "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - \
            Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.")]

    """

    cache_dir = os.path.join(root, "datasets", "AG_NEWS")
    column_names = ["label", "text"]
    datasets_list = []
    path_list = []
    if isinstance(split, str):
        path, _ = cache_file(None, url=URL[split], cache_dir=cache_dir, md5sum=MD5[split], proxies=proxies)
        path_list.append(path)
    else:
        for s in split:
            path, _ = cache_file(None, url=URL[s], cache_dir=cache_dir, md5sum=MD5[s], proxies=proxies)
            path_list.append(path)
    for path in path_list:
        datasets_list.append(GeneratorDataset(source=Agnews(path), column_names=column_names, shuffle=shuffle))
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list


@process.register
def AG_NEWS_Process(dataset, vocab=None, tokenizer=BasicTokenizer(), bucket_boundaries=None,
                    batch_size=512, max_len=500, column="text", drop_remainder=False):
    """
    the process of the AG_News dataset

    Args:
        dataset (GeneratorDataset): AG_News dataset.
        column (str): the column needed to be transpormed of the agnews dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.
        drop_remainder (bool): When the last batch of data contains a data entry smaller than batch_size, whether
            to discard the batch and not pass it to the next operation. Default: False.

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.
        - **Vocab** (Vocab) - vocab created from dataset

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> from mindnlp.dataset import AG_NEWS, AG_NEWS_Process
        >>> train_dataset, test_dataset = AG_NEWS()
        >>> column = "text"
        >>> tokenizer = BasicTokenizer()
        >>> agnews_dataset, vocab = AG_NEWS_Process(train_dataset, column, tokenizer)
        >>> agnews_dataset = agnews_dataset.create_tuple_iterator()
        >>> print(next(agnews_dataset))
        {'label': Tensor(shape=[], dtype=String, value= '3'), 'text': Tensor(shape=[35],
        dtype=Int32, value= [  462,   503,     2,  2102, 47615,  1228,  1766,     3,  1388,
        17,    34,    18,    34,     5,  4076,     5, 10244,     4,   462,   434,    19,    13,
        14141,    21,  3547,     8,  8356,     5, 38127,     4,    55,  4770,  2987,   390,     2])}

    """

    non_str = '\\'
    text_greater = '>'
    text_less = '<'
    str_html = re.compile(r'<[^>]+>')

    for data in dataset:
        src_data = data[1]
        src_data = src_data.asnumpy().tolist()
        if non_str in src_data:
            src_data = src_data.replace(non_str, ' ')
        src_data = html.unescape(src_data)
        if text_less in src_data and text_greater in src_data:
            src_data = str_html.sub('', src_data)

        bows_token = list(src_data)
        data[1] = bows_token

    dataset = dataset.map([tokenizer], 'text')

    if vocab is None:
        vocab = text.Vocab.from_dataset(dataset, columns=column, special_tokens=["<pad>", "<unk>"])
    pad_value = vocab.tokens_to_ids('<pad>')

    lookup_op = text.Lookup(vocab, unknown_token='<unk>')
    type_cast_op = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map([lookup_op], 'text')
    dataset = dataset.map([type_cast_op], 'label')

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")

        trancate_op = TruncateSequence(max_len)
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
