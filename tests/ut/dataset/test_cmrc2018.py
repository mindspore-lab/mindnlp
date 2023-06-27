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
Test cmrc2018
"""

import os
import shutil
import unittest
import pytest
from mindnlp import load_dataset
from mindnlp.dataset import CMRC2018


class Testcmrc2018(unittest.TestCase):
    r"""
    Test cmrc2018
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_cmrc2018(self):
        """Test cmrc2018"""
        num_lines = {
            "train": 10142,
            "validation": 3219,
            "test": 1002,
        }
        dataset_train, dataset_validation, dataset_test = CMRC2018(root=self.root, split=('train', 'validation', 'test'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]
        assert dataset_test.get_dataset_size() == num_lines["test"]
        dataset_train = CMRC2018(root=self.root, split='train')
        dataset_validation = CMRC2018(root=self.root, split='validation')
        dataset_test = CMRC2018(root=self.root, split='test')

        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]
        assert dataset_test.get_dataset_size() == num_lines["test"]


    @pytest.mark.download
    def test_cmrc2018_by_register(self):
        """test cmrc2018 by register"""
        _ = load_dataset('cmrc2018',
                 root=self.root,
                 split=('validation')
                 )
