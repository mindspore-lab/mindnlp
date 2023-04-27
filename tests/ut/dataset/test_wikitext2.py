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
import shutil
import unittest
import pytest
from mindnlp import load_dataset
from mindnlp.dataset import WikiText2


class TestWikiText2(unittest.TestCase):
    r"""
    Test WikiText2
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_wikitext2(self):
        """Test WikiText2"""
        num_lines = {
            "train": 36718,
            "valid": 3760,
            "test": 4358,
        }
        dataset_train, dataset_valid, dataset_test = WikiText2(root=self.root,
                                                               split=('train', 'valid', 'test'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = WikiText2(root=self.root, split='train')
        dataset_valid = WikiText2(root=self.root, split='valid')
        dataset_test = WikiText2(root=self.root, split='test')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_squad2_by_register(self):
        """test squad2 by register"""
        _ = load_dataset('WikiText2',
                 root=self.root,
                 split=('train', 'valid', 'test')
                 )
