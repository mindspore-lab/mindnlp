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
Test QNLI
"""
import os
import unittest
from mindnlp.dataset import QNLI
from mindnlp.dataset import load

class TestQNLI(unittest.TestCase):
    r"""
    Test QNLI
    """

    def setUp(self):
        self.input = None

    def test_qnli(self):
        """Test qnli"""
        num_lines = {
            "train": 104743,
            "dev": 5463,
            "test": 5463,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train, dataset_dev, dataset_test = QNLI(
            root=root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = QNLI(root=root, split="train")
        dataset_dev = QNLI(root=root, split="dev")
        dataset_test = QNLI(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    def test_qnli_by_register(self):
        """test qnli by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load(
            "QNLI",
            root=root,
            split=("train", "dev", "test"),
        )
