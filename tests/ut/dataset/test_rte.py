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
Test RTE
"""
import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import RTE, RTE_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestRTE(unittest.TestCase):
    r"""
    Test RTE
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_rte(self):
        """Test rte"""
        num_lines = {
            "train": 2490,
            "dev": 277,
            "test": 3000,
        }
        dataset_train, dataset_dev, dataset_test = RTE(
            root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = RTE(root=self.root, split="train")
        dataset_dev = RTE(root=self.root, split="dev")
        dataset_test = RTE(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_rte_by_register(self):
        """test rte by register"""
        _ = load_dataset(
            "RTE",
            root=self.root,
            split=("train", "dev", "test"),
        )

    @pytest.mark.download
    def test_rte_process(self):
        r"""
        Test RTE_Process
        """

        train_dataset, _, _ = RTE()
        train_dataset, vocab = RTE_Process(train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32
        assert (next(train_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    def test_rte_process_by_register(self):
        """test rte process by register"""
        train_dataset, _, _ = RTE()
        train_dataset, vocab = process('RTE',
                                dataset=train_dataset,
                                column=("sentence1", "sentence2"),
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32
        assert (next(train_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
