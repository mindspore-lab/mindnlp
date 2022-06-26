# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
test common.data
"""

from collections import Counter
import pandas as pd
import pytest

from mindtext.common import Vocabulary, Pad


TEXT = ["MindText", "works", "well", "in", "most", "cases", "and", "scales", "well", "in",
        "works", "well", "in", "most", "cases", "scales", "well"]
COUNTER = Counter(TEXT)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_add():
    '''this function is test function of add word'''
    vocab = Vocabulary()
    for word in TEXT:
        vocab.add(word)
    assert vocab.word_count == COUNTER

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_update():
    '''this function is test update function'''
    vocab = Vocabulary()
    vocab.update(TEXT)
    vocab.build_vocab()
    assert vocab.word_count == COUNTER

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_to_word():
    '''this function is test idx to word'''
    vocab = Vocabulary()
    word_list = "this is a word list".split()
    vocab.update(word_list)
    vocab.build_vocab()
    assert vocab.to_word(5) == "word"

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_word_to_index():
    '''this function is test word to idx'''
    vocab = Vocabulary()
    word_list = "this is a word list".split()
    vocab.update(word_list)
    vocab.build_vocab()
    word_list = pd.Series([word_list])
    res = vocab.word_to_idx(word_list)
    assert len((res)) == 1

@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_collate():
    '''this function is test collate'''
    pad = Pad(max_length=10)
    input_data = [1, 2, 3, 4, 5, 6]
    result = pad(input_data)
    print(result)
    results = [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]
    assert result == results
