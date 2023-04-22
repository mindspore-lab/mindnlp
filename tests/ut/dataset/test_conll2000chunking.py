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
Test CoNLL2000Chunking
"""
import os
import shutil
import unittest
import pytest
import mindspore as ms
from mindspore.dataset import text
from mindnlp.dataset import CoNLL2000Chunking, CoNLL2000Chunking_Process
from mindnlp import load_dataset, process


class TestCoNLL2000Chunking(unittest.TestCase):
    r"""
    Test CoNLL2000Chunking
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_conll2000chunking(self):
        """Test CoNLL2000Chunking"""
        num_lines = {
            "train": 8936,
            "test": 2012,
        }
        dataset_train, dataset_test = CoNLL2000Chunking(
            root=self.root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = CoNLL2000Chunking(root=self.root, split="train")
        dataset_test = CoNLL2000Chunking(root=self.root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_conll2000chunking_by_register(self):
        """test conll2000chunking by register"""
        _ = load_dataset(
            "CoNLL2000Chunking",
            root=self.root,
            split=("train", "test"),
        )


    @pytest.mark.download
    def test_conll2000chunking_process(self):
        r"""
        Test CoNLL2000Chunking_Process
        """

        test_dataset = CoNLL2000Chunking(split='test')
        vocab = text.Vocab.from_dataset(test_dataset,columns=["words"],freq_range=None,top_k=None,
                                   special_tokens=["<pad>","<unk>"],special_first=True)
        agnews_dataset = CoNLL2000Chunking_Process(test_dataset,vocab)

        agnews_dataset = agnews_dataset.create_tuple_iterator()
        assert (next(agnews_dataset)[1]).dtype == ms.int64


    @pytest.mark.download
    def test_conll2000chunking_process_by_register(self):
        """test CoNLL2000Chunking process by register"""
        test_dataset = CoNLL2000Chunking(split='test')
        vocab = text.Vocab.from_dataset(test_dataset,columns=["words"],freq_range=None,top_k=None,
                                   special_tokens=["<pad>","<unk>"],special_first=True)
        test_dataset = process('CoNLL2000Chunking', test_dataset, vocab)

        test_dataset = test_dataset.create_tuple_iterator()
        assert (next(test_dataset)[1]).dtype == ms.int64
