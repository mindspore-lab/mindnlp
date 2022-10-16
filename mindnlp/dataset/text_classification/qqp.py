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
QQP load function
"""
# pylint: disable=C0103

import os
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT

URL = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"

MD5 = "b6d5672bd9dc1e66ab2bb020ebeafb8d"

class Qqp:
    """
    QQP dataset source
    """

    def __init__(self, path):
        self.path = path
        self._label, self._question1, self._question2 = [], [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        lines.pop(0)
        lines.pop(len(lines) - 1)
        tmp_list = []
        for line in lines:
            l = line.split("\t")
            if len(tmp_list) !=0:
                tmp_list, l = l, tmp_list
                l[-1] += tmp_list[0]
                for i in range(1,len(tmp_list)):
                    l.append(tmp_list[i])
            if len(l)==6:
                self._label.append(int(l[5]))
                self._question1.append(l[3])
                self._question2.append(l[4])
                tmp_list = []
            else:
                tmp_list = l

    def __getitem__(self, index):
        return self._label[index], self._question1[index], self._question2[index]

    def __len__(self):
        return len(self._label)

@load.register
def QQP(root: str = DEFAULT_ROOT, proxies=None):
    r"""
    Load the QQP dataset
    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Examples:
        >>> dataset_train = QQP()
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    cache_dir = os.path.join(root, "datasets", "QQP")
    column_names = ["label", "question1", "question2"]
    path, _ = cache_file(None, url=URL, cache_dir=cache_dir, md5sum=MD5, proxies=proxies)
    return GeneratorDataset(source=Qqp(path), column_names=column_names, shuffle=False)
