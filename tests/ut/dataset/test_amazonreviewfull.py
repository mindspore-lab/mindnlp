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
Test AmazonReviewFull
"""
import os
import unittest
from mindnlp.dataset import AmazonReviewFull
from mindnlp.dataset import load


class TestAmazonReviewFull(unittest.TestCase):
    r"""
    Test AmazonReviewFull
    """

    def setUp(self):
        self.input = None

    def test_amazonreviewfull(self):
        """Test amazonreviewfull"""
        num_lines = {
            "train": 3000000,
            "test": 650000,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train, dataset_test = AmazonReviewFull(
            root=root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = AmazonReviewFull(root=root, split="train")
        dataset_test = AmazonReviewFull(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    def test_amazonreviewfull_by_register(self):
        """test amazonreviewfull by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load(
            "AmazonReviewFull",
            root=root,
            split=("train", "test"),
        )
