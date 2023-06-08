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
import zipfile
import numpy as np
import datasets
import requests

import mindspore
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset, transforms

from mindnlp.vocab import Vocab
from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load_dataset, process
from mindnlp.transforms import BasicTokenizer, Lookup
from mindnlp.configs import DEFAULT_ROOT




URL = "https://bj.bcebos.com/paddlenlp/datasets/DuConv.zip"       

def download_and_extract_zip(url, file_path):
    # Download the zip file
    response = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(response.content)
    
    # Extract the contents of the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(file_path))
    
    # Remove the downloaded zip file
    os.remove(file_path)

# Specify the URL of the zip file to download
zip_url = URL 

# Specify the file path to save the downloaded zip file and extract its contents
zip_file_path = "/home/frank6200/project_mindspore/dataset/DuConv.zip"
# Redefine the zip_file_path
# Download and extract the zip file
download_and_extract_zip(zip_url, zip_file_path)

class Duconv():
    
    """
    DUconV dataset source
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
        with open(self.path, 'r',encoding="utf-8") as f:
            key=0
            for line in f:
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
    
def DuConv(root, split: Union[Tuple[str], str] = ('train', 'dev','test_1','test_2'), proxies=None):
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
        >>> split=('train.txt', 'dev.txt','test_1.txt','test_2.txt')
        >>> dataset_train, dataset_dev,dataset_test_1, dataset_test_2 = DUconV(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))
         
    '''

    file_list = []
    datasets_list = []
    
    if isinstance(split, str):
       split = split.split()
  

    for s in split:
          file_list.append(os.path.join(root,"datasets","DuConv",s))

    for _, file in enumerate(file_list):
            dataset = GeneratorDataset(source =Duconv(file),\
                                   column_names=["id" ,"goal", "knowledge", "conversation", "history","response"],\
                                   shuffle=False)
            datasets_list.append(dataset)
    if len(file_list) == 1:
            return datasets_list[0]
    return datasets_list
