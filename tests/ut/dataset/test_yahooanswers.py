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
Test YahooAnswers
"""
import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import YahooAnswers, YahooAnswers_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestYahooAnswers(unittest.TestCase):
    r"""
    Test YahooAnswers
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_yahooanswers(self):
        """Test yahooanswers"""
        num_lines = {
            "train": 1400000,
            "test": 60000,
        }
        dataset_train, dataset_test = YahooAnswers(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = YahooAnswers(root=self.root, split="train")
        dataset_test = YahooAnswers(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    @pytest.mark.local
    def test_yahooanswers_by_register(self):
        """test yahooanswers by register"""
        _ = load_dataset(
            "YahooAnswers",
            root=self.root,
            split=("train", "test"),
        )

    @pytest.mark.download
    @pytest.mark.local
    def test_yahooanswers_process(self):
        r"""
        Test YahooAnswers_Process
        """

        train_dataset, _ = YahooAnswers()
        train_dataset, vocab = YahooAnswers_Process(train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    @pytest.mark.local
    def test_yahooanswers_process_by_register(self):
        """test yahooanswers process by register"""
        train_dataset, _ = YahooAnswers()
        train_dataset, vocab = process('YahooAnswers',
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
