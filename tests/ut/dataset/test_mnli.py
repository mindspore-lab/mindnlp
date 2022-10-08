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
Test MNLI
"""
import os
import unittest
from mindnlp.dataset import MNLI
from mindnlp.dataset import load


class TestMNLI(unittest.TestCase):
    r"""
    Test MNLI
    """

    def setUp(self):
        self.input = None

    def test_mnli(self):
        """Test mnli"""
        num_lines = {
            "train": 392702,
            "dev_matched": 9815,
            "dev_mismatched": 9832,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        root = "./lijiaming/data"
        dataset_train, dataset_dev_matched, dataset_dev_mismatched = MNLI(
            root=root, split=("train", "dev_matched", "dev_mismatched")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev_matched.get_dataset_size() == num_lines["dev_matched"]
        assert dataset_dev_mismatched.get_dataset_size() == num_lines["dev_mismatched"]

        dataset_train = MNLI(root=root, split="train")
        dataset_dev_matched = MNLI(root=root, split="dev_matched")
        dataset_dev_mismatched = MNLI(root=root, split="dev_mismatched")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev_matched.get_dataset_size() == num_lines["dev_matched"]
        assert dataset_dev_mismatched.get_dataset_size() == num_lines["dev_mismatched"]

    def test_mnli_by_register(self):
        """test mnli by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load(
            "MNLI",
            root=root,
            split=("train", "dev_matched", "dev_mismatched"),
        )
