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
"""Test Vocab"""

from mindnlp import Vocab

def test_vocab_from_list():
    """test vocab from list."""
    vocab = Vocab(['a', 'b', 'c'])

    assert len(vocab) == 3
    assert vocab('a') == 0
    assert 'b' in vocab
    assert vocab['c'] == 2

def test_vocab_with_special_tokens():
    """test vocab with special tokens"""
    vocab = Vocab(['a', 'b', 'c'], special_tokens=['<pad>', '<unk>'])

    assert len(vocab) == 5
    assert vocab('a') == 2
    assert 'b' in vocab
    assert vocab['c'] == 4

def test_vocab_with_special_tokens_last():
    """vocab with special tokens last"""
    vocab = Vocab(['a', 'b', 'c'], special_tokens=['<pad>', '<unk>'], special_first=False)

    assert len(vocab) == 5
    assert vocab('a') == 0
    assert 'b' in vocab
    assert vocab['c'] == 2

def test_vocab_call():
    """vocab call"""
    vocab = Vocab(['a', 'b', 'c'], special_tokens=['<pad>', '<unk>'])
    assert vocab('a') == 2
    assert vocab(2) == 'a'

def test_vocab_lookup():
    """vocab lookup"""
    vocab = Vocab(['a', 'b', 'c', 'd', 'e'], special_tokens=['<pad>', '<unk>'])
    assert vocab.lookup_tokens([2, 1, 3]) == ['a', '<unk>', 'b']
    assert vocab.lookup_ids(['b', 'd', 'e']) == [3, 5, 6]

def test_append_token():
    """test append token"""
    vocab = Vocab(['a', 'b', 'c'], special_tokens=['<pad>', '<unk>'])
    vocab.append_token('d')

    assert 'd' in vocab
    assert vocab['d'] == 5
