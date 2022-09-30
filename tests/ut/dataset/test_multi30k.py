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
import unittest
import mindspore
from mindspore.dataset import text
from mindnlp.dataset import Multi30k, Multi30k_Process
from mindnlp.dataset import load, process


class TestMulti30k(unittest.TestCase):
    r"""
    Test Multi30k
    """

    def setUp(self):
        self.input = None

    def test_multi30k(self):
        """Test Multi30k"""
        num_lines = {
            "train": 29000,
            "valid": 1014,
            "test": 1000,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_valid, dataset_test = Multi30k(root=root,
                                                              split=(
                                                                  'train', 'valid', 'test'),
                                                              language_pair=(
                                                                  'de', 'en')
                                                              )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = Multi30k(
            root=root, split='train', language_pair=('de', 'en'))
        dataset_valid = Multi30k(
            root=root, split='valid', language_pair=('en', 'de'))
        dataset_test = Multi30k(
            root=root, split='test', language_pair=('de', 'en'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    def test_multi30k_by_register(self):
        """test multi30k by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('multi30k',
                 root=root,
                 split=('train', 'valid', 'test'),
                 language_pair=('de', 'en')
                 )


class TestMulti30kProcess(unittest.TestCase):
    r"""
    Test Multi30K Process
    """

    def setUp(self):
        self.input = None

    def test_multi30k_process_no_vocab(self):
        r"""
        Test multi30k process with no vocab
        """

        test_dataset = Multi30k(
            root="./dataset",
            split="test",
            language_pair=("de", "en")
        )

        test_dataset, vocab = Multi30k_Process(
            test_dataset, text.BasicTokenizer(), "en")

        for i in test_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    def test_multi30k_process_no_vocab_by_register(self):
        '''
        Test multi30k process with no vocab by register
        '''

        test_dataset = Multi30k(
            root="./dataset",
            split="test",
            language_pair=("de", "en")
        )

        test_dataset, vocab = process('Multi30k', test_dataset)

        for i in test_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
