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
Test Multi30k
"""
import os
import shutil
import unittest
import pytest
import mindspore
from mindspore.dataset import text
from mindnlp.dataset import Multi30k, Multi30k_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestMulti30k(unittest.TestCase):
    r"""
    Test Multi30k
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    def test_multi30k(self):
        """Test Multi30k"""
        num_lines = {
            "train": 29000,
            "valid": 1014,
            "test": 1000,
        }
        dataset_train, dataset_valid, dataset_test = Multi30k(root=self.root,
                                                              split=(
                                                                  'train', 'valid', 'test'),
                                                              language_pair=(
                                                                  'de', 'en')
                                                              )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = Multi30k(
            root=self.root, split='train', language_pair=('de', 'en'))
        dataset_valid = Multi30k(
            root=self.root, split='valid', language_pair=('en', 'de'))
        dataset_test = Multi30k(
            root=self.root, split='test', language_pair=('de', 'en'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.download
    def test_multi30k_by_register(self):
        """test multi30k by register"""
        _ = load_dataset('multi30k',
                 root=self.root,
                 split=('train', 'valid', 'test'),
                 language_pair=('de', 'en')
                 )

    @pytest.mark.download
    def test_multi30k_process(self):
        r"""
        Test multi30k process
        """

        train_dataset = Multi30k(
            root=self.root,
            split="train",
            language_pair=("de", "en")
        )

        tokenizer = BasicTokenizer(True)
        train_dataset = train_dataset.map([tokenizer], 'en')
        train_dataset = train_dataset.map([tokenizer], 'de')

        en_vocab = text.Vocab.from_dataset(train_dataset, 'en', special_tokens=['<pad>', '<unk>'], special_first= True)
        de_vocab = text.Vocab.from_dataset(train_dataset, 'de', special_tokens=['<pad>', '<unk>'], special_first= True)

        vocab = {'en':en_vocab, 'de':de_vocab}

        train_dataset = Multi30k_Process(train_dataset, vocab=vocab)

        for i in train_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in en_vocab.vocab().items():
            assert isinstance(value, int)
            break

        for _, value in de_vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.download
    def test_multi30k_process_by_register(self):
        '''
        Test multi30k process by register
        '''

        train_dataset = Multi30k(
            root=self.root,
            split="train",
            language_pair=("de", "en")
        )

        tokenizer = BasicTokenizer(True)
        train_dataset = train_dataset.map([tokenizer], 'en')
        train_dataset = train_dataset.map([tokenizer], 'de')

        en_vocab = text.Vocab.from_dataset(train_dataset, 'en', special_tokens=['<pad>', '<unk>'], special_first= True)
        de_vocab = text.Vocab.from_dataset(train_dataset, 'de', special_tokens=['<pad>', '<unk>'], special_first= True)

        vocab = {'en':en_vocab, 'de':de_vocab}

        train_dataset = process('Multi30k', train_dataset, vocab = vocab)

        for i in train_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in en_vocab.vocab().items():
            assert isinstance(value, int)
            break

        for _, value in de_vocab.vocab().items():
            assert isinstance(value, int)
            break
