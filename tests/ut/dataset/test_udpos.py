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
Test UDPOS
"""
import os
import shutil
import unittest
import pytest
from mindnlp.dataset import UDPOS
from mindnlp.dataset import load_dataset


class TestUDPOS(unittest.TestCase):
    r"""
    Test UDPOS
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_udpos(self):
        """Test UDPOS"""
        num_lines = {
            "train": 12543,
            "dev": 2002,
            "test": 2077,
        }
        dataset_train, dataset_dev, dataset_test = UDPOS(
            root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = UDPOS(root=self.root, split="train")
        dataset_dev = UDPOS(root=self.root, split="dev")
        dataset_test = UDPOS(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_udpos_by_register(self):
        """test udpos by register"""
        _ = load_dataset(
            "UDPOS",
            root=self.root,
            split=("train", "dev", "test"),
        )
