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
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import MNLI, MNLI_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestMNLI(unittest.TestCase):
    r"""
    Test MNLI
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_mnli(self):
        """Test mnli"""
        num_lines = {
            "train": 392702,
            "dev_matched": 9815,
            "dev_mismatched": 9832,
        }
        dataset_train, dataset_dev_matched, dataset_dev_mismatched = MNLI(
            root=self.root, split=("train", "dev_matched", "dev_mismatched")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev_matched.get_dataset_size() == num_lines["dev_matched"]
        assert dataset_dev_mismatched.get_dataset_size() == num_lines["dev_mismatched"]

        dataset_train = MNLI(root=self.root, split="train")
        dataset_dev_matched = MNLI(root=self.root, split="dev_matched")
        dataset_dev_mismatched = MNLI(root=self.root, split="dev_mismatched")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev_matched.get_dataset_size() == num_lines["dev_matched"]
        assert dataset_dev_mismatched.get_dataset_size() == num_lines["dev_mismatched"]

    @pytest.mark.download
    @pytest.mark.local
    def test_mnli_by_register(self):
        """test mnli by register"""
        _ = load_dataset(
            "MNLI",
            root=self.root,
            split=("dev_matched", "dev_mismatched"),
        )

    @pytest.mark.download
    @pytest.mark.local
    def test_mnli_process(self):
        r"""
        Test MNLI_Process
        """

        dev_dataset = MNLI(split='dev_matched')
        dev_dataset, vocab = MNLI_Process(dev_dataset)

        dev_dataset = dev_dataset.create_tuple_iterator()
        assert (next(dev_dataset)[1]).dtype == ms.int32
        assert (next(dev_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    @pytest.mark.local
    def test_mnli_process_by_register(self):
        """test mnli process by register"""
        dev_dataset = MNLI(split='dev_matched')
        dev_dataset, vocab = process('MNLI',
                                dataset=dev_dataset,
                                column=("sentence1", "sentence2"),
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        dev_dataset = dev_dataset.create_tuple_iterator()
        assert (next(dev_dataset)[1]).dtype == ms.int32
        assert (next(dev_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
