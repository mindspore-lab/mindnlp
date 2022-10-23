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
Test transforms
"""
import unittest
import numpy as np
from mindspore.dataset import GeneratorDataset
from mindnlp.dataset.transforms import BasicTokenizer

class TestBasicTokenizer(unittest.TestCase):
    """
    test BasicTokenizer
    """
    def setUp(self):
        self.test_str = "I am making small mistakes during working hours"
        self.texts = [
            'Welcome to Beijing!',
            '北京欢迎您！',
            '我喜欢China!',
        ]

    def test_do_lower_case(self):
        """test do lower case on eager mode"""
        tokenizer = BasicTokenizer(lower_case=True)
        output = tokenizer(self.test_str)
        assert output.dtype.type is np.str_

    def test_dataset_map(self):
        """test dataset map"""
        test_dataset = GeneratorDataset(self.texts, 'text')
        test_dataset = test_dataset.map(BasicTokenizer())
        print(next(test_dataset.create_tuple_iterator()))
