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
IMDB dataset
"""
# pylint: disable=C0103

import os
import re
import string
import tarfile
from typing import Union, Tuple
import six
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT

URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

MD5 = "7c2ac02c03563afcf9b574c7e56c153a"


class Imdb:
    """
    IMDB dataset source
    """
    label_map = {
        "pos": 1,
        "neg": 0
    }

    def __init__(self, path, mode) -> None:
        self.path = path
        self.mode: str = mode
        self._label, self._text = [], []
        self._load("pos")
        self._load("neg")

    def _load(self, label):
        pattern = re.compile(fr"aclImdb/{self.mode}/{label}/.*\.txt$")
        with tarfile.open(self.path) as tarf:
            tf = tarf.next()
            while tf is not None:
                if bool(pattern.match(tf.name)):
                    self._text.append(str(tarf.extractfile(tf).read().rstrip(six.b("\n\r"))
                                      .translate(None, six.b(string.punctuation)).lower()).split())
                    self._label.append(self.label_map[label])
                tf = tarf.next()

    def __getitem__(self, index):
        return self._label[index], self._text[index]

    def __len__(self):
        return len(self._label)


@load.register
def IMDB(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train", "test"),
    proxies=None,
):
    r"""
    Load the IMDB dataset
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
        >>> dataset_train,dataset_test = IMDB()
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    cache_dir = os.path.join(root, "datasets", "IMDB")
    column_names = ["label", "text"]
    mode_list = []
    datasets_list = []
    cache_file(
        None,
        cache_dir=cache_dir,
        url=URL,
        md5sum=MD5,
        proxies=proxies,
    )
    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)
    for mode in mode_list:
        datasets_list.append(
            GeneratorDataset(
                source=Imdb(os.path.join(cache_dir,"aclImdb_v1.tar.gz"), mode),
                column_names=column_names, shuffle=False
            )
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list
