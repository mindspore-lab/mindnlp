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
Test DuConv
"""

import os
import shutil
import unittest
import pytest
from mindnlp import load_dataset
from mindnlp.dataset import hf_duconv



class TestDuConv(unittest.TestCase):
    r"""
    Test DuConv 
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_duconv(self):
        """Test DuConv"""
        num_lines = {
            "train": 19900,
            "dev": 2000,
            "test1":5000,
            "test2":10100,
        }
        dataset_train, dataset_dev, dataset_test1, dataset_test2 = hf_duconv(root=self.root, split=('train', 'dev','test1','test2'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test1.get_dataset_size() == num_lines["test1"]
        assert dataset_test2.get_dataset_size() == num_lines["test2"]

        dataset_train = hf_duconv(root=self.root, split='train')
        dataset_dev = hf_duconv(root=self.root, split='dev')
        dataset_test1 = hf_duconv(root=self.root, split='test1')
        dataset_test2 = hf_duconv(root=self.root, split='test2')

        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test1.get_dataset_size() == num_lines["test1"]
        assert dataset_test2.get_dataset_size() == num_lines["test2"]

    @pytest.mark.download
    def test_duconv_by_register(self):
        """test hf_duconv by register"""
        _ = load_dataset('hf_duconv', root=self.root, split='dev')
        