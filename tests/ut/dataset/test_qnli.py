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
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import QNLI, QNLI_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer

class TestQNLI(unittest.TestCase):
    r"""
    Test QNLI
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_qnli(self):
        """Test qnli"""
        num_lines = {
            "train": 104743,
            "dev": 5463,
            "test": 5463,
        }
        dataset_train, dataset_dev, dataset_test = QNLI(
            root=self.root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = QNLI(root=self.root, split="train")
        dataset_dev = QNLI(root=self.root, split="dev")
        dataset_test = QNLI(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    @pytest.mark.local
    def test_qnli_by_register(self):
        """test qnli by register"""
        _ = load_dataset(
            "QNLI",
            root=self.root,
            split=("dev", "test"),
        )

    @pytest.mark.download
    @pytest.mark.local
    def test_qnli_process(self):
        r"""
        Test QNLI_Process
        """

        test_dataset = QNLI(split='test')
        test_dataset, vocab = QNLI_Process(test_dataset)

        test_dataset = test_dataset.create_tuple_iterator()
        assert (next(test_dataset)[0]).dtype == ms.int32
        assert (next(test_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    @pytest.mark.local
    def test_qnli_process_by_register(self):
        """test qnli process by register"""
        test_dataset = QNLI(split='test')
        test_dataset, vocab = process('QNLI',
                                dataset=test_dataset,
                                column=("question", "sentence"),
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        print(test_dataset.get_col_names())
        test_dataset = test_dataset.create_tuple_iterator()
        assert (next(test_dataset)[0]).dtype == ms.int32
        assert (next(test_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
