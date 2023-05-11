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
"""Test the BertTokenizer"""

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.text import Vocab as msVocab
from mindnlp import Vocab
from mindnlp.transforms import BertTokenizer

def test_bert_tokenizer_mindnlp_vocab():
    """test BertTokenizer by dataset.map"""
    texts = ['i make a small mistake when i\'m working! 床前明月光']
    test_dataset = GeneratorDataset(texts, 'text')
    vocab_list = ["床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低",
              "思", "故", "乡","繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak",
              "make", "small", "mistake", "##s", "during", "work", "##ing", "hour", "😀", "😃",
              "😄", "😁", "+", "/", "-", "=", "12", "28", "40", "16", " ", "I", "[CLS]", "[SEP]",
              "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]
    vocab = Vocab(vocab_list)
    bert_tokenizer = BertTokenizer(vocab=vocab, lower_case=True, return_token=True)
    test_dataset = test_dataset.map(operations=bert_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 19
    assert dataset_after.dtype == ms.string

def test_bert_tokenizer_mindspore_vocab():
    """test BertTokenizer by dataset.map"""
    texts = ['i make a small mistake when i\'m working! 床前明月光']
    test_dataset = GeneratorDataset(texts, 'text')
    vocab_list = ["床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低",
              "思", "故", "乡","繁", "體", "字", "嘿", "哈", "大", "笑", "嘻", "i", "am", "mak",
              "make", "small", "mistake", "##s", "during", "work", "##ing", "hour", "😀", "😃",
              "😄", "😁", "+", "/", "-", "=", "12", "28", "40", "16", " ", "I", "[CLS]", "[SEP]",
              "[UNK]", "[PAD]", "[MASK]", "[unused1]", "[unused10]"]
    vocab = msVocab.from_list(vocab_list)
    bert_tokenizer = BertTokenizer(vocab=vocab, lower_case=True, return_token=True)
    test_dataset = test_dataset.map(operations=bert_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 19
    assert dataset_after.dtype == ms.string

def test_bert_tokenizer_from_pretrained():
    """test BertTokenizer from pretrained."""
    texts = ['i make a small mistake when i\'m working! 床前明月光']
    test_dataset = GeneratorDataset(texts, 'text')

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', return_token=True)
    test_dataset = test_dataset.map(operations=bert_tokenizer)
    dataset_after = next(test_dataset.create_tuple_iterator())[0]

    assert len(dataset_after) == 21
    assert dataset_after.dtype == ms.string


def test_bert_tokenizer_add_special_tokens():
    """test add special tokens."""
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    cls_id = bert_tokenizer.token_to_id("[CLS]")

    assert cls_id is not None
