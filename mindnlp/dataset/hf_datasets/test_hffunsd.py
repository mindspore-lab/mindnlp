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
Test FUNSD
"""
import os
import shutil
import unittest
import pytest
from mindnlp.dataset import HF_FUNSD
from mindnlp import load_dataset


class TestFUNSD(unittest.TestCase):
    r"""
    Test funsd
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_funsd(self):
        """Test funsd"""
        num_lines = {
            "train": 149,
            "test": 50,
        }
        dataset_train, dataset_test = HF_FUNSD(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = HF_FUNSD(root=self.root, split="train")
        dataset_test = HF_FUNSD(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    @pytest.mark.local
    def test_funsd_by_register(self):
        """test xfund by register"""
        _ = load_dataset(
            "HF_FUNSD",
            root=self.root,
            split=("train", "test"),
        )
