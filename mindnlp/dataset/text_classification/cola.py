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
CoLA load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load, process
from mindnlp.dataset.process import common_process
from mindnlp.dataset.transforms import BasicTokenizer
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import unzip

URL = "https://nyu-mll.github.io/CoLA/cola_public_1.1.zip"

MD5 = "9f6d88c3558ec424cd9d66ea03589aba"


class Cola:
    """
    CoLA dataset source
    """

    def __init__(self, path) -> None:
        self.path: str = path
        self._source, self._label, self._sentence = [], [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        if not self.path.endswith("out_of_domain_dev.tsv"):
            lines.pop(len(lines) - 1)
        for line in lines:
            l = line.split("\t")
            self._source.append(l[0])
            self._label.append(l[1])
            self._sentence.append(l[-1])

    def __getitem__(self, index):
        return self._source[index], self._label[index], self._sentence[index]

    def __len__(self):
        return len(self._sentence)


@load.register
def CoLA(
    root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "dev", "test"), proxies=None
):
    r"""
    Load the CoLA dataset

    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'dev', 'test').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'dev', 'test')
        >>> dataset_train,dataset_dev,dataset_test = CoLA(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= 'gj04'), Tensor(shape=[], dtype=String, \
        \value= '1'), \Tensor(shape=[], dtype=String, value= "Our friends won't buy \
        this analysis, let alone the \next one we propose.")]
    """
    cache_dir = os.path.join(root, "datasets", "CoLA")
    path_dict = {
        "train": "in_domain_train.tsv",
        "dev": "in_domain_dev.tsv",
        "test": "out_of_domain_dev.tsv",
    }
    column_names = ["source", "label", "sentence"]
    path_list = []
    datasets_list = []
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, proxies=proxies)
    unzip(path, cache_dir)
    if isinstance(split, str):
        path_list.append(
            os.path.join(cache_dir, "cola_public", "raw", path_dict[split])
        )
    else:
        for s in split:
            path_list.append(
                os.path.join(cache_dir, "cola_public", "raw", path_dict[s])
            )
    for path in path_list:
        datasets_list.append(
            GeneratorDataset(
                source=Cola(path), column_names=column_names, shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def CoLA_Process(dataset, column="sentence", tokenizer=BasicTokenizer(), vocab=None):
    """
    the process of the CoLA dataset

    Args:
        dataset (GeneratorDataset): CoLA dataset.
        column (str): the column needed to be transpormed of the CoLA dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.
        - **Vocab** (Vocab) - vocab created from dataset

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> from mindnlp.dataset import CoLA, CoLA_Process
        >>> train_dataset, dataset_dev, dataset_test  = CoLA()
        >>> column = "sentence"
        >>> tokenizer = BasicTokenizer()
        >>> train_dataset, vocab = CoLA_Process(train_dataset, column, tokenizer)
        >>> train_dataset = train_dataset.create_tuple_iterator()
        >>> print(next(train_dataset))
        [Tensor(shape=[], dtype=String, value= 'gj04'), Tensor(shape=[], dtype=String, value= '1'),
        Tensor(shape=[17], dtype=Int32, value= [ 854,  290,  196,   10,   28,  182,   57,  738,    9,
        816, 1372,    1,  768,   99,   71, 5316,    0])]
    """

    return common_process(dataset, column, tokenizer, vocab)
