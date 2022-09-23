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
Test Multi30k
"""

import unittest
from mindnlp.dataset.multi30k import Multi30k


class TestMulti30k(unittest.TestCase):
    r"""
    Test Multi30k
    """

    def setUp(self):
        self.input = None

    def test_multi30k(self):
        """Test Multi30k"""
        num_lines = {
            "train": 29000,
            "valid": 1014,
            "test": 1000,
        }
        train_dataset, valid_dataset, test_dataset = Multi30k(root=r"./dataset",
                                                              split=(
                                                                  'train', 'valid', 'test'),
                                                              language_pair=(
                                                                  'de', 'en')
                                                              )
        assert train_dataset.get_dataset_size() == num_lines["train"]
        assert valid_dataset.get_dataset_size() == num_lines["valid"]
        assert test_dataset.get_dataset_size() == num_lines["test"]

        train_dataset = Multi30k(
            root=r"./dataset", split='train', language_pair=('de', 'en'))
        valid_dataset = Multi30k(
            root=r"./dataset", split='valid', language_pair=('en', 'de'))
        test_dataset = Multi30k(
            root=r"./dataset", split='test', language_pair=('de', 'en'))
        assert train_dataset.get_dataset_size() == num_lines["train"]
        assert valid_dataset.get_dataset_size() == num_lines["valid"]
        assert test_dataset.get_dataset_size() == num_lines["test"]
