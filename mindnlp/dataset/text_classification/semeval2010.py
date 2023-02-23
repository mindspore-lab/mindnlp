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
SemEval2010_Task8 dataset
"""
# pylint: disable=C0103

import os
import json
import numpy as np
from typing import Union, Tuple
import mindspore
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.register import load, process
from mindnlp.configs import DEFAULT_ROOT

embedding_download_path = "http://metaoptimize.s3.amazonaws.com/hlbl-embeddings-ACL2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt.gz"
dataset_download_path = "https://github.com/onehaitao/CNN-relation-extraction/data" # 经过处理的数据集，原github库位置

class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self):
        self.path_word = '../../configs/semeval2010/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt'  # path of pre-trained word embedding
        self.word_dim = 50  # dimension of word embedding

    def load_embedding(self):
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character

        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        pad_emb = np.zeros([1, self.word_dim], dtype=np.float32)  # <pad> is initialize as zero
        word_vec = np.concatenate((pad_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        # word_vec = torch.from_numpy(word_vec)
        word_vec =Tensor.from_numpy(word_vec)
        return word2id, word_vec


class RelationLoader(object):
    """
    A loader for relation2id
    """
    def __init__(self):
        self.data_dir = '../../configs/semeval2010'

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


class SemEval2010():
    """
    SemEval2010 dataset source
    """
    def __init__(self, filename, rel2id, word2id):
        self.filename = filename
        self.rel2id = rel2id
        self.word2id = word2id
        self.max_len = 100
        self.pos_dis = 50
        self.data_dir = '../../configs/semeval2010'
        self.dataset, self.label = self.__load_data()

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def __get_relative_pos(self, x, entity_pos):
        if x < entity_pos[0]:
            return self.__get_pos_index(x-entity_pos[0])
        elif x > entity_pos[1]:
            return self.__get_pos_index(x-entity_pos[1])
        else:
            return self.__get_pos_index(0)

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                e1_pos (tuple) span of e1
                e2_pos (tuple) span of e2
                sentence (list)
        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['*UNKNOWN*']))
            pos1.append(self.__get_relative_pos(i, e1_pos))
            pos2.append(self.__get_relative_pos(i, e2_pos))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

                pos1.append(self.__get_relative_pos(i, e1_pos))
                pos2.append(self.__get_relative_pos(i, e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['sentence']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


@load.register
def SemEval(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ("train.json", "test.json"),
):
    r"""
    Load the SemEval2010_Task8 dataset

    Args:
        root (str): Directory where the datasets are saved.
            Default:~/.mindnlp
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train.json', 'test.json').

    Returns:
        - **datasets_list** (list) -A list of loaded datasets.
          If only one type of dataset is specified,such as 'trian',
          this dataset is returned instead of a list of datasets.

    Examples:
        >>> root = "~/.mindnlp"
        >>> split = ('train.json', 'test.json')
        >>> dataset_train,dataset_test = SemEval(root, split)
        >>> train_iter = dataset_train.create_tuple_iterator()
        >>> print(next(train_iter))

    """

    column_names = ["data", "label"]
    mode_list = []
    datasets_list = []
    shuffle_list = {"train.json":True, "test.json":False}
    if isinstance(split, str):
        mode_list.append(split)
    else:
        for s in split:
            mode_list.append(s)
    word2id, word_vec = WordEmbeddingLoader().load_embedding()
    rel2id, id2rel, class_num = RelationLoader().get_relation()
    for mode in mode_list:
        datasets_list.append(
            GeneratorDataset(
                source=SemEval2010(mode, rel2id, word2id),
                column_names=column_names, shuffle=shuffle_list[mode]
            )
        )
    if len(mode_list) == 1:
        return datasets_list[0]
    return datasets_list, word_vec, class_num

@process.register
def SemEval_Process(dataset, batch_size=64, drop_remainder=False):
    """
    the process of the SemEval2010_Task8 dataset

    Args:
        dataset (GeneratorDataset): SemEval2010_Task8 dataset.
        batch_size (int): size of the batch.
        drop_remainder (bool): If True, will drop the last batch for each bucket if it is not a full batch

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
    >>> semeval, word_vec, class_num = load('semeval')
    >>> semeval_train, semeval_test = semeval
    >>> semeval_train = process('semeval', semeval_train, batch_size=batch_size, drop_remainder = True)
    """

    def concat(data):
        data = np.concatenate(data, axis=0)
        return data
    dataset = dataset.map(operations=concat, input_columns="data")

    def asarr(label):
        label = np.asarray(label, dtype=np.int32)
        return label
    dataset = dataset.map(operations=asarr, input_columns="label")

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    return dataset
