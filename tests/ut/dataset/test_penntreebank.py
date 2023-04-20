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
Test PennTreebank
"""

import os
import shutil
import unittest
import pytest
from mindnlp import load_dataset
from mindnlp.dataset import PennTreebank

class TestPennTreebank(unittest.TestCase):
    r"""
    Test PennTreebank
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_penn_treebank(self):
        """Test PennTreebank"""
        num_lines = {
            "train": 42068,
            "valid": 3370,
            "test": 3761,
        }
        dataset_train, dataset_valid, dataset_test = PennTreebank(root=self.root,
                                                                  split=('train', 'valid', 'test'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = PennTreebank(root=self.root, split='train')
        dataset_valid = PennTreebank(root=self.root, split='valid')
        dataset_test = PennTreebank(root=self.root, split='test')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_penn_treebank_by_register(self):
        """test penn treebank by register"""
        _ = load_dataset('PennTreebank',
                 root=self.root,
                 split=('train', 'valid', 'test')
                 )
