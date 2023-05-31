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
Test Ptb_text_only
"""
import os
import shutil
import unittest
import pytest
from mindnlp.dataset import HF_Ptb_text_only, HF_Ptb_text_only_Process
from mindnlp import load_dataset, process
from mindnlp.transforms import BasicTokenizer


class TestPtbTextOnly(unittest.TestCase):
    r"""
    Test Ptb_text_only
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_ptb_text_only(self):
        """Test ptb_text_only"""
        num_lines = {
            "train": 42068,
            "validation": 3370,
            "test": 3761,
        }
        dataset_train, dataset_dev, dataset_test = HF_Ptb_text_only(
            root=self.root, split=("train", "validation", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["validation"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = HF_Ptb_text_only(root=self.root, split="train")
        dataset_dev = HF_Ptb_text_only(root=self.root, split="validation")
        dataset_test = HF_Ptb_text_only(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["validation"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_ptb_text_only_by_register(self):
        """test ptb_text_only by register"""
        _ = load_dataset(
            "HF_Ptb_text_only",
            root=self.root,
            split=("train", "validation", "test"),
        )

    @pytest.mark.download
    def test_ptb_text_only_process(self):
        r"""
        Test HF_Ptb_text_only_Process
        """

        train_dataset, _, _ = HF_Ptb_text_only()
        train_dataset, vocab = HF_Ptb_text_only_Process(train_dataset)
        train_dataset = train_dataset.create_tuple_iterator()

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    def test_ptb_text_only_process_by_register(self):
        """test ptb_text_only process by register"""
        train_dataset, _, _ = HF_Ptb_text_only()
        train_dataset, vocab = process('HF_Ptb_text_only',
                                       dataset=train_dataset,
                                       column="sentence",
                                       tokenizer=BasicTokenizer(),
                                       vocab=None
                                       )
        train_dataset = train_dataset.create_tuple_iterator()

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
