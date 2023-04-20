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
Test YelpReviewFull
"""
import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import YelpReviewFull, YelpReviewFull_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestYelpReviewFull(unittest.TestCase):
    r"""
    Test YelpReviewFull
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_yelpreviewfull(self):
        """Test yelpreviewfull"""
        num_lines = {
            "train": 650000,
            "test": 50000,
        }
        dataset_train, dataset_test = YelpReviewFull(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = YelpReviewFull(root=self.root, split="train")
        dataset_test = YelpReviewFull(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    @pytest.mark.local
    def test_yelpreviewfull_by_register(self):
        """test yelpreviewfull by register"""
        _ = load_dataset(
            "YelpReviewFull",
            root=self.root,
            split=("train", "test"),
        )

    @pytest.mark.download
    def test_yelpreviewfull_process(self):
        r"""
        Test YelpReviewFull_Process
        """

        train_dataset, _ = YelpReviewFull()
        train_dataset, vocab = YelpReviewFull_Process(train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    @pytest.mark.local
    def test_yelpreviewfull_process_by_register(self):
        """test yelpreviewfull process by register"""
        train_dataset, _ = YelpReviewFull()
        train_dataset, vocab = process('YelpReviewFull',
                                dataset=train_dataset,
                                column="title_text",
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
