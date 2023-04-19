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
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import AmazonReviewFull, AmazonReviewFull_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestAmazonReviewFull(unittest.TestCase):
    r"""
    Test AmazonReviewFull
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_amazonreviewfull(self):
        """Test amazonreviewfull"""
        num_lines = {
            "train": 3000000,
            "test": 650000,
        }
        dataset_train, dataset_test = AmazonReviewFull(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = AmazonReviewFull(root=self.root, split="train")
        dataset_test = AmazonReviewFull(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    @pytest.mark.local
    def test_amazonreviewfull_by_register(self):
        """test amazonreviewfull by register"""
        _ = load_dataset(
            "AmazonReviewFull",
            root=self.root,
            split="test"
        )

    @pytest.mark.download
    @pytest.mark.local
    def test_amazonreviewfull_process(self):
        r"""
        Test AmazonReviewFull_Process
        """

        test_dataset = AmazonReviewFull(split='test')
        amazonreviewfull_dataset, vocab = AmazonReviewFull_Process(test_dataset)

        amazonreviewfull_dataset = amazonreviewfull_dataset.create_tuple_iterator()
        assert (next(amazonreviewfull_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    @pytest.mark.local
    def test_amazonreviewfull_process_by_register(self):
        """test amazonreviewfull process by register"""
        test_dataset = AmazonReviewFull(split='test')
        test_dataset, vocab = process('AmazonReviewFull',
                                dataset=test_dataset,
                                column="title_text",
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        test_dataset = test_dataset.create_tuple_iterator()
        assert (next(test_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
