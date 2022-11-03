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
WNLI dataset
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset, text
from mindnlp.utils.download import cache_file
from mindnlp.dataset.process import common_process
from mindnlp.dataset.register import load, process
from mindnlp.dataset.transforms import BasicTokenizer
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import unzip

URL = "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip"

MD5 = "a1b4bd2861017d302d29e42139657a42"


class Wnli:
    """
    WNLI dataset source
    """

    def __init__(self, path) -> None:
        self.path: str = path
        self._label,self._sentence1,self._sentence2 = [],[],[]
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        lines.pop(0)
        lines.pop(len(lines)-1)
        if self.path.endswith("test.tsv"):
            for line in lines:
                l = line.split('\t')
                self._sentence1.append(l[1])
                self._sentence2.append(l[2])
        else:
            for line in lines:
                l = line.split('\t')
                self._sentence1.append(l[1])
                self._sentence2.append(l[2])
                self._label.append(l[3])


    def __getitem__(self,index):
        if self.path.endswith("test.tsv"):
            return self._sentence1[index],self._sentence2[index]
        return self._label[index],self._sentence1[index],self._sentence2[index]

    def __len__(self):
        return len(self._sentence1)


@load.register
def WNLI(
    root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "dev", "test"), proxies=None
):
    r"""
    Load the WNLI dataset

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
        >>> split = ("train", "dev, "test")
        >>> dataset_train,dataset_dev,dataset_test = WNLI(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= '1'), Tensor(shape=[], dtype=String,
        value= 'I stuck a pin through a carrot. When I pulled the pin out, it had a hole.'),
        Tensor(shape=[], dtype=String, value= 'The carrot had a hole.')]

    """
    cache_dir = os.path.join(root, "datasets", "WNLI")
    path_dict = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv",
    }
    column_names_dict = {
        "train": ["label","sentence1","sentence2"],
        "dev": ["label","sentence1","sentence2"],
        "test": ["sentence1","sentece2"],
    }
    column_names = []
    path_list = []
    datasets_list = []
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, proxies=proxies)
    unzip(path, cache_dir)
    if isinstance(split, str):
        path_list.append(
            os.path.join(cache_dir, "WNLI", path_dict[split])
        )
        column_names.append(column_names_dict[split])
    else:
        for s in split:
            path_list.append(
                os.path.join(cache_dir, "WNLI", path_dict[s])
            )
            column_names.append(column_names_dict[s])
    for idx, path in enumerate(path_list):
        datasets_list.append(
            GeneratorDataset(
                source=Wnli(path), column_names=column_names[idx], shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def WNLI_Process(dataset,
    column: Union[Tuple[str], str] = ("sentence1", "sentence2"),
    tokenizer=BasicTokenizer(),
    vocab=None
):
    """
    the process of the WNLI dataset

    Args:
        dataset (GeneratorDataset): WNLI dataset.
        column (Tuple[str]|str): the column or columns needed to be transpormed of the WNLI dataset
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset
        vocab (Vocab): vocabulary object, used to store the mapping of token and index

    Returns:
        - **dataset** (MapDataset) - dataset after transforms
        - **Vocab** (Vocab) - vocab created from dataset

    Raises:
        TypeError: If `column` is not a string or Tuple[str]

    Examples:
        >>> from mindnlp.dataset import WNLI, WNLI_Process
        >>> dataset_train, dataset_dev, dataset_test= WNLI()
        >>> dataset_train, vocab = WNLI_Process(dataset_train)
        >>> dataset_train = dataset_train.create_tuple_iterator()
        >>> print(next(dataset_train))
        [Tensor(shape=[], dtype=String, value= '1'), Tensor(shape=[20],
        dtype=Int32, value= [  23, 1102,    6,  341,  109,    6,  607,    0,  105,   23,  468,
        1,  341,   33,    2,    9,   14,    6,  182,    0]), Tensor(shape=[6], dtype=Int32,
        value= [  7, 607,  14,   6, 182,   0]

    """

    if isinstance(column, str):
        return common_process(dataset, column, tokenizer, vocab)
    if vocab is None:
        for col in column:
            dataset = dataset.map(tokenizer, input_columns=col)
        column = list(column)
        vocab = text.Vocab.from_dataset(dataset, columns=column)
        for col in column:
            dataset = dataset.map(text.Lookup(vocab), input_columns=col)
        return dataset, vocab
    for col in column:
        dataset = dataset.map(tokenizer, input_columns=col)
    for col in column:
        dataset = dataset.map(text.Lookup(vocab), input_columns=col)
    return dataset, vocab
