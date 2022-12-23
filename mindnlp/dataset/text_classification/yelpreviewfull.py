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
YelpReviewFull load function
"""
# pylint: disable=C0103

import os
import csv
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load, process
from mindnlp.dataset.process import common_process
from mindnlp.dataset.transforms import BasicTokenizer
from mindnlp.configs import DEFAULT_ROOT
from mindnlp.utils import untar

URL = "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0&confirm=t"

MD5 = "f7ddfafed1033f68ec72b9267863af6c"


class Yelpreviewfull:
    """
    YelpReviewFull dataset source
    """

    def __init__(self, path) -> None:
        self.path: str = path
        self._label, self._title_text = [], []
        self._load()

    def _load(self):
        csvfile = open(self.path, "r", encoding="utf-8")
        dict_reader = csv.reader(csvfile)
        for row in dict_reader:
            self._label.append(int(row[0]))
            self._title_text.append(f"{row[1]}")

    def __getitem__(self, index):
        return self._label[index], self._title_text[index]

    def __len__(self):
        return len(self._label)


@load.register
def YelpReviewFull(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "test"),
    proxies=None,
):
    r"""
    Load the YelpReviewFull dataset

    Args:
        root (str): Directory where the datasets are saved.
            Default:'~/.mindnlp'
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train', 'test').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ("train", "test")
        >>> dataset_train,dataset_test = YelpReviewFull(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    cache_dir = os.path.join(root, "datasets", "YelpReviewFull")
    path_dict = {
        "train": "train.csv",
        "test": "test.csv",
    }
    column_names = ["label", "title_text"]
    path_list = []
    datasets_list = []
    path, _ = cache_file(
        None,
        cache_dir=cache_dir,
        url=URL,
        md5sum=MD5,
        download_file_name="yelp_review_full_csv.tar.gz",
        proxies=proxies,
    )

    untar(path, cache_dir)
    if isinstance(split, str):
        path_list.append(os.path.join(
            cache_dir, "yelp_review_full_csv", path_dict[split]))
    else:
        for s in split:
            path_list.append(os.path.join(
                cache_dir, "yelp_review_full_csv", path_dict[s]))
    for path in path_list:
        datasets_list.append(
            GeneratorDataset(
                source=Yelpreviewfull(path), column_names=column_names, shuffle=False
            )
        )
    if len(path_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def YelpReviewFull_Process(dataset, column="title_text", tokenizer=BasicTokenizer(), vocab=None):
    """
    the process of the YelpReviewFull dataset

    Args:
        dataset (GeneratorDataset): YelpReviewFull dataset.
        column (str): the column needed to be transpormed of the YelpReviewFull dataset.
        tokenizer (TextTensorOperation): tokenizer you choose to tokenize the text dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.
        - **Vocab** (Vocab) - vocab created from dataset

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
        >>> from mindnlp.dataset import YelpReviewFull, YelpReviewFull_Process
        >>> train_dataset, dataset_test  = YelpReviewFull()
        >>> column = "sentence"
        >>> tokenizer = BasicTokenizer()
        >>> train_dataset, vocab = YelpReviewFull_Process(train_dataset, column, tokenizer)
        >>> train_dataset = train_dataset.create_tuple_iterator()
        >>> print(next(train_dataset))
        {'label': Tensor(shape=[], dtype=Int64, value= 5), 'title_text': Tensor(shape=[117],
        dtype=Int32, value= [  6338,      0, 258139,   1500,    265,    139,    295,     12,     15,
        6,   1344,  17531,      0,    101,      8,     28,    106,      3,    702,     7,    842,      7,
        364,    199,  11063,    277,    101,      8,     28,    152,     25,     57,     15,   1076,
        225,   4021,    277,    101,      8,     28,  12202,     19,      6,    308,     20,   1638,   3077,
        43, 287710,     38,     76,     23,   1802,     27,   1151,      7,     44,     14,     53,   1617,
        15,    852,    185,   1865,      3,    21,    248,   3990,    277,      3,     21,     67,     52,
        16374,      7,    169,  19483,    364,    390,      7,    169,    279,  138,      0,     75,      2,
        79,     81,    103,     21,    248,     63,    139,      8,     99,    570,     51,    387,      7,
        143,     10,    155,   1532,    139,     27,     64,    279,      2,     18,    139,      8,     99,
        75,   9730,      6,   6598,      0])}

    """

    return common_process(dataset, column, tokenizer, vocab)
