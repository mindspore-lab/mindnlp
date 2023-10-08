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
Test imdb
"""

import unittest
import pytest
from mindnlp.dataset import load_dataset


class TestIMDB(unittest.TestCase):
    r"""
    Test imdb with huggingface datasets
    """

    @pytest.mark.download
    @pytest.mark.local
    def test_load_single(self):
        """Test HF_Docvqa_zh"""
        dataset = load_dataset('imdb', split='train')
        print(dataset)
        print(next(dataset.create_tuple_iterator()))


    @pytest.mark.download
    @pytest.mark.local
    def test_load_single_with_tuple(self):
        """Test HF_Docvqa_zh"""
        dataset = load_dataset('imdb', split=('train',))
        print(dataset)
        print(next(dataset.create_tuple_iterator()))

    @pytest.mark.download
    @pytest.mark.local
    def test_load_single_with_list(self):
        """Test HF_Docvqa_zh"""
        dataset = load_dataset('imdb', split=['train'])
        print(dataset)
        print(next(dataset.create_tuple_iterator()))

    @pytest.mark.download
    @pytest.mark.local
    def test_load_single_with_dict(self):
        """Test HF_Docvqa_zh"""
        dataset = load_dataset('imdb', split=None)
        print(dataset.keys())
        print(next(dataset['train'].create_tuple_iterator()))

    @pytest.mark.download
    @pytest.mark.local
    def test_load_single_with_streaming(self):
        """Test HF_Docvqa_zh"""
        dataset = load_dataset('imdb', split=None, streaming=True)
        print(dataset.keys())
        print(next(dataset['train'].create_tuple_iterator()))

    @pytest.mark.download
    @pytest.mark.local
    def test_split(self):
        """Test HF_Docvqa_zh"""
        dataset = load_dataset('imdb', split='train')
        train, _ = dataset.split([0.7, 0.3])
        print(next(train.create_tuple_iterator()))
