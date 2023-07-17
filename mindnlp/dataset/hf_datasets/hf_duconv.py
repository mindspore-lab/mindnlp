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
DUcONV load function
"""
import os
import json
from typing import Tuple, Union

from mindspore.dataset import GeneratorDataset, transforms
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load_dataset, process

from mindnlp.configs import DEFAULT_ROOT
from mindnlp.transforms import Lookup, Truncate
from mindnlp.utils import unzip

from mindnlp.dataset.utils import make_bucket

URL = "https://bj.bcebos.com/paddlenlp/datasets/DuConv.zip"

class Duconv:
    """
    Duconv dataset source
    """
    def __init__(self, path) -> None:
        self.path = path
        self._id = []
        self._goal = []
        self._knowledge= []
        self._conversation = []
        self._history = []
        self._response = []
        self._load()

    def _load(self):
        with open(self.path, 'r',encoding="utf-8") as f_file:
            key=0
            for line in f_file:
                json_data = json.loads(line)
                duconv = json_data

                goal = duconv.get("goal", [[]])
                knowledge = duconv.get("knowledge", [[]])
                conversation = duconv.get("conversation", [])
                history = duconv.get("history", [])
                response = duconv.get("response", "")

                self._goal.append(goal)
                self._knowledge.append(knowledge)
                self._conversation.append(conversation)
                self._history.append(history)
                self._response.append(response)
                self._id.append(key)
                key += 1

    def __getitem__(self, index):
        return self._id[index], self._goal[index], self._knowledge[index],\
            self._conversation[index], self._history[index], self._response[index]

    def __len__(self):
        return len(self._response)

@load_dataset.register
def hf_duconv(root:str=DEFAULT_ROOT, \
              split: Union[Tuple[str], str] = ('train', 'dev','test_1','test_2'),proxies=None):
    r'''
    Load the DuConv dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train','dev').
        proxies (dict): a dict to identify proxies,for example: {"https": "https://127.0.0.1:7890"}.

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Raises:
        TypeError: If `root` is not a string.
        TypeError: If `split` is not a string or Tuple[str].

    Examples:
        >>> root = "~/.mindnlp"
        >>> split=('train', 'dev','test_1','test_2')
        >>> dataset_train, dataset_dev,dataset_test_1, dataset_test_2 = hf_duconv(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
         
    '''

    mode_list = []
    datasets_list = []
    cache_dir=os.path.join(root, "datasets", "DuConv")
    if isinstance(split, str):
        mode_list.append(cache_dir + '/DuConv/' + split + '.txt')
    file_path, _ = cache_file(None, cache_dir=cache_dir, url=URL,
                              download_file_name="DuConv.zip", proxies=proxies)
    unzip(file_path,cache_dir)
    for split_item in split:
        mode_list.append(cache_dir + '/DuConv/' + split_item + '.txt')
    for _, every_ds in enumerate(mode_list):
        dataset = GeneratorDataset(source = Duconv(every_ds),\
        column_names=["id" ,"goal", "knowledge", "conversation", "history","response"],\
                                   shuffle=False)
        datasets_list.append(dataset)
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def hf_duconv_process(dataset, tokenizer, vocab, batch_size=64, max_context_len=1000, max_qa_len=30,
                      bucket_boundaries=None, drop_remainder=False)->None:
    """
    the process of the duconv dataset

    """
    pad_value = vocab.tokens_to_ids('<pad>')
    lookup_op = Lookup(vocab, unk_token='<unk>')

    dataset = dataset.map([tokenizer, lookup_op], 'goal')
    dataset = dataset.map([tokenizer, lookup_op], 'knowledge')
    dataset = dataset.map([tokenizer, lookup_op], 'conversation')

    pad_qa_op = transforms.PadEnd([max_qa_len], pad_value)
    dataset = dataset.map([pad_qa_op], 'knowledge')
    dataset = dataset.map([pad_qa_op], 'conversation')

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        trancate_context_op = Truncate(max_context_len)
        dataset = dataset.map([trancate_context_op], 'goal')
        trancate_qa_op = Truncate(max_qa_len)
        dataset = dataset.map([trancate_qa_op], 'knowledge')
        dataset = dataset.map([trancate_qa_op], 'conversation')

        if bucket_boundaries[-1] < max_context_len + 1:
            bucket_boundaries.append(max_context_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = make_bucket(dataset, 'goal', pad_value,
                              bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_context_op = transforms.PadEnd([max_context_len], pad_value)
        dataset = dataset.map([pad_context_op], 'goal')

        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    return dataset
