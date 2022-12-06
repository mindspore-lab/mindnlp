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
import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset, text, transforms

from mindnlp.utils.download import cache_file
from mindnlp.dataset.register import load, process
from mindnlp.dataset.transforms import BasicTokenizer
from mindnlp.configs import DEFAULT_ROOT

URL = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MD5 = {
    "train": "981b29407e0affa3b1b156f72073b945",
    "dev": "3e85deb501d4e538b6bc56f786231552",
}


class Squad1():
    """
    SQuAD1 dataset source
    """

    def __init__(self, path):
        self.path = path
        self._id = []
        self._context = []
        self._question = []
        self._anwsers = []
        self._s_idex = []
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding='utf8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragragh in article['paragraphs']:
                    context = paragragh['context']
                    for qa in paragragh['qas']:
                        ids = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            self._id.append(ids)
                            self._context.append(context)
                            self._question.append(question)
                            answer = ans['text']
                            self._anwsers.append(answer)
                            s_idx = ans['answer_start']
                            self._s_idex.append(s_idx)
                            break

    def __getitem__(self, index):
        return self._id[index], self._context[index], self._question[index],\
            self._anwsers[index], self._s_idex[index]

    def __len__(self):
        return len(self._anwsers)


@load.register
def SQuAD1(
    root: str = DEFAULT_ROOT,
    split: Union[Tuple[str], str] = ('train', 'dev'),
    shuffle=False,
    proxies=None
):
    r"""
    Load the SQuAD1 dataset

    Args:
        root (str): Directory where the datasets are saved.
        split (str|Tuple[str]): Split or splits to be returned.
            Default:('train','dev').
        shuffle (bool): Whether or not to perform shuffle on the dataset.
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
                                       "id" ,"context", "question", "answers", "answer_start"],
                                   shuffle=shuffle)
        datasets_list.append(dataset)
    if len(file_list) == 1:
        return datasets_list[0]
    return datasets_list

@process.register
def SQuAD1_Process(dataset, char_vocab, word_vocab=None,\
                   tokenizer=BasicTokenizer(True),\
                   max_context_len=768, max_question_len=64, max_char_len=48,\
                   batch_size=64, drop_remainder=False):
    """
    the process of the squad1 dataset

    Args:
        dataset (GeneratorDataset): Squad1 dataset.
        tokenizer (TextTensorOperation): Tokenizer you choose to tokenize the text dataset.
        word_vocab (Vocab): Vocabulary object of words, used to store the mapping of the token and index.
        char_vocab (Vocab): Vocabulary object of chars, used to store the mapping of the token and index.
        max_context_len (int): Max length of the context. Default: 768.
        max_question_len (int): Max length of the question. Default: 64.
        max_char_len (int): Max length of the char. Default: 48.
        batch_size (int): The number of rows each batch is created with. Default: 64.
        drop_remainder (bool): When the last batch of data contains a data entry smaller than batch_size, whether
            to discard the batch and not pass it to the next operation. Default: False.

    Returns:
        - MapDataset, Squad1 Dataset after process.

    Raises:
        TypeError: If `word_vocab` is not of type text.Vocab.
        TypeError: If `char_vocab` is not of type text.Vocab.
        TypeError: If `max_context_len` is not of type int.
        TypeError: If `max_question_len` is not of type int.
        TypeError: If `max_char_len` is not of type int.
        TypeError: If `batch_size` is not of type int.
        TypeError: If `drop_remainder` is not of type bool.

    Examples:
        >>> from mindspore.dataset import text
        >>> from mindnlp.dataset import SQuAD1, SQuAD1_Process
        >>> char_dic = {"<unk>": 0, "<pad>": 1, "e": 2, "t": 3, "a": 4, "i": 5, "n": 6,\
                    "o": 7, "s": 8, "r": 9, "h": 10, "l": 11, "d": 12, "c": 13, "u": 14,\
                    "m": 15, "f": 16, "p": 17, "g": 18, "w": 19, "y": 20, "b": 21, ",": 22,\
                    "v": 23, ".": 24, "k": 25, "1": 26, "0": 27, "x": 28, "2": 29, "\"": 30, \
                    "-": 31, "j": 32, "9": 33, "'": 34, ")": 35, "(": 36, "?": 37, "z": 38,\
                    "5": 39, "8": 40, "q": 41, "3": 42, "4": 43, "7": 44, "6": 45, ";": 46,\
                    ":": 47, "\u2013": 48, "%": 49, "/": 50, "]": 51, "[": 52}
        >>> char_vocab = text.Vocab.from_dict(char_dic)
        >>> dev_dataset = SQuAD1(split='dev')
        >>> squad_dev = SQuAD1_Process(dataset=dev_dataset, char_vocab=char_vocab)
        >>> squad_dev = squad_dev.create_tuple_iterator()
        >>> print(next(squad_dev))
    """

    c_char_list = []
    q_char_list = []
    c_lens = []
    q_lens = []
    s_idx = []
    e_idx = []
    pad_value_char = char_vocab.tokens_to_ids('<pad>')
    abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']
    for data in dataset:
        context = data[1].asnumpy().tolist()
        question = data[2].asnumpy().tolist()
        answer = data[3].asnumpy().tolist()
        c_token = tokenizer(context)
        c_len = len(c_token)
        q_token = tokenizer(question)
        q_len = len(q_token)
        s_index = int(data[4])
        e_index = s_index + len(answer)
        c_char = []
        q_char = []
        # find the starting and ending position of the answer
        l = 0
        s_found = False
        for i, token in enumerate(c_token):
            while l < len(context):
                if context[l] in abnormals:
                    l += 1
                else:
                    break

            l += len(token)
            if l > s_index and s_found is False:
                s_index = i
                s_found = True
            if l >= e_index:
                e_index = i
                break
        # generate the char list of the context(after lookup and padding operation)
        for token in c_token:
            token_ids = char_vocab.tokens_to_ids(list(token))
            if isinstance(token_ids, int):
                token_list = []
                token_list.append(token_ids)
                token_ids = token_list
            Pad_char = transforms.PadEnd(pad_shape=[max_char_len], pad_value=pad_value_char)
            token_pad = Pad_char(token_ids)
            token_pad = np.array(token_pad, dtype=np.int32)
            c_char.append(token_pad)
        # generate the char list of the question(after lookup and padding operation)
        for token in q_token:
            token_ids = char_vocab.tokens_to_ids(list(token))
            if isinstance(token_ids, int):
            # if type(token_ids)==int:
                token_list = []
                token_list.append(token_ids)
                token_ids = token_list
            Pad_char = transforms.PadEnd(pad_shape=[max_char_len], pad_value=pad_value_char)
            token_pad = Pad_char(token_ids)
            token_pad = np.array(token_pad, dtype=np.int32)
            q_char.append(token_pad)

        c_lens.append(c_len)
        q_lens.append(q_len)
        s_idx.append(s_index)
        e_idx.append(e_index)
        c_char_list.append(c_char)
        q_char_list.append(q_char)

    data = (c_char_list, q_char_list, c_lens, q_lens, s_idx, e_idx)
    dataset2 = ds.NumpySlicesDataset(data=data, column_names=["c_char", "q_char", "c_lens",\
                                     "q_lens", "s_idx", "e_idx"], shuffle=False)

    dataset = dataset.zip(dataset2)
    dataset = dataset.rename(input_columns="id", output_columns="ids")
    columns_to_project = ["ids", "context", "question", "c_char", "q_char", "c_lens", "q_lens", "s_idx", "e_idx"]
    dataset = dataset.project(columns=columns_to_project)

    dataset = dataset.map(tokenizer, 'context', 'c_word')
    dataset = dataset.map(tokenizer, 'question', 'q_word')

    if word_vocab is None:
        word_vocab = text.Vocab.from_dataset(dataset, columns=['c_word', 'q_word'],\
                                             special_tokens=["<unk>", "<pad>"], special_first=True)

    lookup_op = text.Lookup(word_vocab, unknown_token='<unk>')
    type_cast_op = transforms.TypeCast(mindspore.int32)
    pad_value_word = word_vocab.tokens_to_ids('<pad>')

    dataset = dataset.map(lookup_op, 'c_word')
    dataset = dataset.map(lookup_op, 'q_word')
    dataset = dataset.map(type_cast_op, 'c_lens')
    dataset = dataset.map(type_cast_op, 'q_lens')
    dataset = dataset.map(type_cast_op, 's_idx')
    dataset = dataset.map(type_cast_op, 'e_idx')

    pad_op_context = transforms.PadEnd([max_context_len], pad_value_word)
    dataset = dataset.map([pad_op_context], 'c_word')
    pad_op_question = transforms.PadEnd([max_question_len], pad_value_word)
    dataset = dataset.map([pad_op_question], 'q_word')
    pad_char_context = transforms.PadEnd([max_context_len, max_char_len], pad_value_word)
    dataset = dataset.map([pad_char_context], 'c_char')
    pad_char_question = transforms.PadEnd([max_question_len, max_char_len], pad_value_word)
    dataset = dataset.map([pad_char_question], 'q_char')

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset
