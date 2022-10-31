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
Test AG_NEWS
"""
import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import AG_NEWS, AG_NEWS_Process
from mindnlp.dataset import load, process
from mindnlp.dataset.transforms import BasicTokenizer


class TestAGNEWS(unittest.TestCase):
    r"""
    Test AG_NEWS
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.dataset
    @pytest.mark.local
    def test_agnews(self):
        """Test agnews"""
        num_lines = {
            "train": 120000,
            "test": 7600,
        }
        dataset_train, dataset_test = AG_NEWS(
            root=self.root, split=("train", "test"))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = AG_NEWS(root=self.root, split="train")
        dataset_test = AG_NEWS(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.dataset
    def test_agnews_by_register(self):
        """test agnews by register"""
        _ = load('AG_NEWS', root=self.root, split='test')

    @pytest.mark.dataset
    def test_agnews_process(self):
        r"""
        Test AG_NEWS_Process
        """

        test_dataset = AG_NEWS(split='test')
        agnews_dataset, vocab = AG_NEWS_Process(test_dataset)

        agnews_dataset = agnews_dataset.create_tuple_iterator()
        assert (next(agnews_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.dataset
    def test_agnews_process_by_register(self):
        """test agnews process by register"""
        test_dataset = AG_NEWS(split='test')
        test_dataset, vocab = process('AG_NEWS',
                                dataset=test_dataset,
                                column="text",
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        test_dataset = test_dataset.create_tuple_iterator()
        assert (next(test_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
