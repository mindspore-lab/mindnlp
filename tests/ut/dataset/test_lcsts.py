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
Test LCSTS
"""

import os
import shutil
import unittest
import pytest
from mindnlp import load_dataset
from mindnlp.dataset import LCSTS


class TestLCSTS(unittest.TestCase):
    r"""
    Test LCSTS
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_lcsts(self):
        """Test LCSTS"""
        num_lines = {
            "train": 1470769,
            "dev": 10666,
        }
        dataset_train, dataset_dev = LCSTS(root=self.root, split=('train', 'dev'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

        dataset_train = LCSTS(root=self.root, split='train')
        dataset_dev = LCSTS(root=self.root, split='dev')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

    @pytest.mark.download
    def test_lcsts_by_register(self):
        """test lcsts by register"""
        _ = load_dataset('lcsts',
                 root=self.root,
                 split='dev'
                 )
