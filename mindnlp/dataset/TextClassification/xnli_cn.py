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
XNLI_CN load function
"""
# pylint: disable=C0103

import os
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset, text
from mindnlp.utils.download import cache_file
from mindnlp.dataset.process import common_process
from mindnlp.dataset.register import load_dataset, process
from mindnlp.transforms import BasicTokenizer
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar, unzip

URL = "https://bj.bcebos.com/paddlenlp/datasets/xnli_cn.tar.gz"

MD5 = "aaf6de381a2553d61d8e6fad4ba96499"


class xnli_cn:
    """
    XNLI_CN dataset source
    __all__ = ["XNLI"]
    ALL_LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

    """
    label_map = {
        "entailment": 0,
        "neutral": 1,
        "contradictory": 2,
    }

    def __init__(self, path) -> None:
        self.path: str = path
        self._sentence1,self._sentence2,self._label = [],[],[]
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines:
            l = line.split('\t')
            if l[0] == '':
                break
            self._sentence1.append(l[0])
            self._sentence2.append(l[1])
            self._label.append(self.label_map[l[2]])


    def __getitem__(self,index):
        return self._sentence1[index],self._sentence2[index],self._label[index]

    def __len__(self):
        return len(self._sentence1)


@load_dataset.register
def XNLI_CN(
    root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ("train", "dev", "test"), proxies=None
):
    r"""
    Load the XNLI dataset

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
        >>> split = ("train", "dev", "test")
        >>> dataset_train, dataset_dev, dataset_test = XNLI_CN(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        [Tensor(shape=[], dtype=String, value= '从概念上看,奶油收入有两个基本方面产品和地理.'), 
        Tensor(shape=[], dtype=String, value= '产品和地理是什么使奶油抹霜工作.'), 
        Tensor(shape=[], dtype=Int64, value= 1)]
    """
    if root == DEFAULT_ROOT:
        cache_dir = os.path.join(root, "datasets","XNLI_CN")
    else:
        cache_dir = root
    path_dict = {
        "train": "train/part-0",
        "dev": "dev/part-0",
        "test": "test/part-0",
    }
    column_names = ["sentence1", "sentence2", "label"]
    column_names_list = []
    path_list = []
    datasets_list = []
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, download_file_name="xnli_cn.tar.gz", proxies=proxies)
    untar(path, cache_dir)
    if isinstance(split, str):
        path_list.append(
            os.path.join(cache_dir, "xnli_cn", path_dict[split])
        )
        column_names_list.append(column_names)
    else:
        for s in split:
            path_list.append(
                os.path.join(cache_dir, "xnli_cn", path_dict[s])
            )
            column_names_list.append(column_names)
    for idx, path in enumerate(path_list):
        datasets_list.append(
            GeneratorDataset(
                source=xnli_cn(path), column_names=column_names_list[idx], shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def XNLI_CN_Process(dataset,
    column: Union[Tuple[str], str] = ("sentence1", "sentence2"),
    tokenizer=BasicTokenizer(),
    vocab=None
):
    """
    the process of the XNLI_CN dataset

    Args:
        dataset (GeneratorDataset): XNLI_CN dataset.
        column (Tuple[str]|str): the column or columns needed to be transpormed of the XNLI_CN dataset
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset
        vocab (Vocab): vocabulary object, used to store the mapping of token and index

    Returns:
        - **dataset** (MapDataset) - dataset after transforms
        - **Vocab** (Vocab) - vocab created from dataset

    Raises:
        TypeError: If `column` is not a string or Tuple[str]

    Examples:
        >>> from mindnlp.dataset import XNLI_CN, XNLI_CN_Process
        >>> dataset_train, dataset_dev, dataset_test= XNLI_CN()
        >>> dataset_train, vocab = XNLI_CN_Process(dataset_train)
        >>> dataset_train = dataset_train.create_tuple_iterator()
        >>> print(next(dataset_train))
        [Tensor(shape=[23], dtype=Int32, 
        value= [  77, 1023,  725,   27,   58,    3, 1343,  922,  211,  203,    9,  190,    
        10,   248,  138,   47,  151,  238,  290,   16,   39,  102,    4]), Tensor(shape=[16], 
        dtype=Int32, value= [ 238,  290,   16,   39,  102,    6,  116,   67,  147, 1343,  
        922,  2274, 3049,   66,   49,    4]), Tensor(shape=[], dtype=Int64, value= 1)]

    """

    if isinstance(column, str):
        return common_process(dataset, column, tokenizer, vocab)
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