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
SQuAD1 load function
"""
# pylint: disable=C0103

import os
import json
from typing import Tuple, Union
from mindspore.dataset import GeneratorDataset
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load
from mindnlp.configs import DEFAULT_ROOT

URL = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MD5 = {
    "train": "981b29407e0affa3b1b156f72073b945",
    "dev": "3e85deb501d4e538b6bc56f786231552",
}


class Squad1:
    """
    SQuAD1 dataset source
    """

    def __init__(self, path):
        self.path = path
        self._context, self._question = [], []
        self._anwsers, self._answers_start = [], []
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding='utf8') as f:
            json_data = json.load(f)
            for i in range(len(json_data["data"])):
                for j in range(len(json_data["data"][i]["paragraphs"])):
                    for k in range(len((json_data["data"][i]["paragraphs"][j]["qas"]))):
                        answers = []
                        answers_start = []
                        self._context.append(
                            json_data["data"][i]["paragraphs"][j]["context"])
                        self._question.append(
                            json_data["data"][i]["paragraphs"][j]["qas"][k]["question"])
                        for index in range(len(json_data["data"][i]
                                               ["paragraphs"][j]["qas"][k]["answers"])):
                            answers.append(json_data["data"][i]["paragraphs"][j]["qas"][k]
                                           ["answers"][index]['text'])
                            answers_start.append(json_data["data"][i]["paragraphs"][j]["qas"][k]
                                                 ["answers"][index]['answer_start'])
                        self._anwsers.append(answers)
                        self._answers_start.append(answers_start)

    def __getitem__(self, index):
        return self._context[index], self._question[index],\
            self._anwsers[index], self._answers_start[index]

    def __len__(self):
        return len(self._question)


@load.register
def SQuAD1(root: str = DEFAULT_ROOT, split: Union[Tuple[str], str] = ('train', 'dev'), proxies=None):
    r"""
    Load the SQuAD1 dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train','dev').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
            If only one type of dataset is specified,such as 'trian',
            this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train', 'dev')
        >>> dataset_train, dataset_dev = SQuAD1(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
        {'context': Tensor(shape=[], dtype=String, value= 'Architecturally, \
            the school has a Catholic character. Atop the Main Building\'s gold dome ...'),
        'question': Tensor(shape=[], dtype=String, value= 'To whom did the Virgin Mary allegedly \
            appear in 1858 in Lourdes France?'),
        'answers': Tensor(shape=[1], dtype=String, value= ['Saint Bernadette Soubirous']),
        'answers_start': Tensor(shape=[1], dtype=Int32, value= [515])}

    """
    cache_dir = os.path.join(root, "datasets", "SQuAD1")
    file_list = []
    datasets_list = []
    if isinstance(split, str):
        split = split.split()
    for s in split:
        path, _ = cache_file(
            None, url=URL[s], cache_dir=cache_dir, md5sum=MD5[s], proxies=proxies
        )
        file_list.append(path)

    for _, file in enumerate(file_list):
        dataset = GeneratorDataset(source=Squad1(file),
                                   column_names=[
                                       "context", "question", "answers", "answers_start"],
                                   shuffle=False)
        datasets_list.append(dataset)
    if len(file_list) == 1:
        return datasets_list[0]
    return datasets_list
