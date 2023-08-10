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
Test SQuAD2
"""

import os
import shutil
import unittest
import pytest

from mindspore.dataset import text
from mindnlp.transforms import BasicTokenizer
from mindnlp.dataset import HF_SQuAD2, HF_SQuAD2_Process


class TestHFSQuAD2(unittest.TestCase):
    r"""
    Test HF_SQuAD2
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_hf_squad2(self):
        """Test HF_SQuAD2"""
        num_lines = {
            "train": 130319,
            "validation": 11873,
        }
        dataset_train, dataset_validation = HF_SQuAD2(root=self.root, split=('train', 'validation'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

        dataset_train = HF_SQuAD2(root=self.root, split='train')
        dataset_validation = HF_SQuAD2(root=self.root, split='validation')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_validation.get_dataset_size() == num_lines["validation"]

    @pytest.mark.download
    def test_hf_squad2_process(self):
        """
        Test hf_squad2 process
        """

        train_dataset = HF_SQuAD2(
            root=self.root,
            split="train"
        )

        tokenizer = BasicTokenizer(True)
        train_dataset = train_dataset.map([tokenizer], 'context')

        vocab = text.Vocab.from_dataset(train_dataset, 'context', special_tokens=['<pad>', '<unk>'],
                                        special_first=True)

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

        train_dataset = HF_SQuAD2(
            root=self.root,
            split="train"
        )
        train_dataset = HF_SQuAD2_Process(train_dataset, tokenizer=tokenizer, vocab=vocab)

        for i in train_dataset.create_tuple_iterator():
            assert i[0].shape == (64, 1000)
            assert i[1].shape == (64, 30)
            assert i[2].shape == (64, 30)
            assert i[3].shape == (64,)
            break

    @pytest.mark.download
    def test_hf_squad2_process_bucket_boundaries(self):
        """
        Test hf_squad2 process with bucket_boundaries
        """

        train_dataset = HF_SQuAD2(
            root=self.root,
            split="train"
        )

        tokenizer = BasicTokenizer(True)
        train_dataset = train_dataset.map([tokenizer], 'context')

        vocab = text.Vocab.from_dataset(train_dataset, 'context', special_tokens=['<pad>', '<unk>'],
                                        special_first=True)

        train_dataset = HF_SQuAD2(
            root=self.root,
            split="train"
        )
        dataset = HF_SQuAD2_Process(train_dataset, tokenizer=tokenizer, vocab=vocab,
                                    bucket_boundaries=[400, 500], max_context_len=800, drop_remainder=True)

        for i in dataset.create_tuple_iterator():
            assert i[0].shape == (64, 400 - 1)
            break
