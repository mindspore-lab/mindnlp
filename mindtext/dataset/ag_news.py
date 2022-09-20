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
AG_NEWS dataset
"""
# pylint: disable=C0103

import csv
from typing import Union, Tuple
from mindspore.dataset import GeneratorDataset
from utils.download import cache_file

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
        self._load()

    def _load(self):
        print(self.path)
        csvfile = open(self.path, "r")
        dict_reader = csv.reader(csvfile)
        for row in dict_reader:
            self._label.append(row[0])
            self._text.append(f"{row[1]} {row[2]}")

    def __getitem__(self, index):
        return self._label[index], self._text[index]

    def __len__(self):
        return len(self._text)


def AG_News(root: str = "./data", split: Union[Tuple[str], str] = ("train", "test")):
    r"""
    If the dataset exists and passes the md5 test, return the dataset, otherwise re-download the dataset and return.

    Args:
        root (str): directory where the datasets are saved
        split (Union[Tuple[str], str]): split or splits to be returned

    Returns:
        - **dataset**(GeneratorDataset)

    Examples:
        >>> ds1,ds2 = dataset.ag_news.AG_News()
        >>> for label,text in ds1:
        >>>     print(label,text)
        >>>     break
        4 Judge Revokes Mine Permit in Florida (AP) AP - A federal judge Friday revoked a permit to develop a
        limestone mine amid 6,000 acres of habitat that could be used by the endangered Florida panther.

    """
    path_list = []
    for s in split:
        path, _ = cache_file(
            None, url=URL[s], cache_dir=f"{root}", md5sum=MD5[s]
        )
        path_list.append(path)
    if len(path_list) == 1:
        return GeneratorDataset(source=Agnews(path_list[0]), column_names=["label", "text"])
    return GeneratorDataset(
        source=Agnews(path_list[0]), column_names=["label", "text"]
    ), GeneratorDataset(source=Agnews(path_list[1]), column_names=["label", "text"])
