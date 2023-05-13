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
Test IMDB
"""
import os
import shutil
import unittest
import pytest
from mindnlp import load_dataset
from mindnlp.dataset import IMDB


class TestIMDB(unittest.TestCase):
    r"""
    Test IMDB
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_imdb(self):
        """Test imdb"""
        num_lines = {
            "train": 25000,
            "test": 25000,
        }
        dataset_train, dataset_test = IMDB(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = IMDB(root=self.root, split="train")
        dataset_test = IMDB(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_imdb_by_register(self):
        """test imdb by register"""
        _ = load_dataset(
            "IMDB",
            root=self.root,
            split=("train", "test"),
        )
