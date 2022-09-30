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
Test WikiText2
"""

import os
import unittest
from mindnlp.dataset import WikiText2, load


class TestWikiText2(unittest.TestCase):
    r"""
    Test WikiText2
    """

    def setUp(self):
        self.input = None

    def test_wikitext2(self):
        """Test WikiText2"""
        num_lines = {
            "train": 36718,
            "valid": 3760,
            "test": 4358,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_valid, dataset_test = WikiText2(root=root,
                                                               split=('train', 'valid', 'test'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = WikiText2(root=root, split='train')
        dataset_valid = WikiText2(root=root, split='valid')
        dataset_test = WikiText2(root=root, split='test')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    def test_squad2_by_register(self):
        """test squad2 by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('WikiText2',
                 root=root,
                 split=('train', 'valid', 'test')
                 )
