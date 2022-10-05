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
Test DBpedia
"""
import os
import unittest
from mindnlp.dataset import DBpedia
from mindnlp.dataset import load


class TestDBpedia(unittest.TestCase):
    r"""
    Test DBpedia
    """

    def setUp(self):
        self.input = None

    def test_dbpedia(self):
        """Test DBpedia"""
        num_lines = {
            "train": 560000,
            "test": 70000,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train, dataset_test = DBpedia(
            root=root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = DBpedia(root=root, split="train")
        dataset_test = DBpedia(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    def test_dbpedia_by_register(self):
        """test dbpedia by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load(
            "dbpedia",
            root=root,
            split=("train", "test"),
        )
