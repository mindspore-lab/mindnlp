# Copyright 2023 Huawei Technologies Co., Ltd
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
SQuAD1 process function
"""
# pylint: disable=C0103

import numpy as np

import mindspore
import mindspore.dataset as ds
from mindspore.dataset import transforms, text

from mindnlp.transforms import BasicTokenizer

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
    abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009','\u200B', '\u0303', '\u092e']
    for data in dataset:
        context = data[1].asnumpy().tolist()
        question = data[2].asnumpy().tolist()
        answer = data[3].asnumpy().tolist()
        c_token = tokenizer(context)
        c_len = len(c_token)
        q_token = tokenizer(question)
        q_len = len(q_token)
        answer_token = tokenizer(answer)
        answer_len = len(answer_token)
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
        # exceptional cases
        if s_index >= c_len or e_index >= c_len:
            for i, token in enumerate(c_token):
                if token == answer_token[0]:
                    s_index = i
                    if c_token[i + answer_len - 1] == answer_token[-1]:
                        e_index = i + answer_len - 1
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
