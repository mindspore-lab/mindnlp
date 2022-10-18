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
Test WikiText2
"""

import os
import unittest
import pytest
import mindspore
from mindspore.dataset import text
from mindnlp.dataset import WikiText2, WikiText2_Process, load, process


class TestWikiText2(unittest.TestCase):
    r"""
    Test WikiText2
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_wikitext2(self):
        """Test WikiText2"""
        num_lines = {
            "train": 36718,
            "valid": 3760,
            "test": 4358,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_valid, dataset_test = WikiText2(root=root,
                                                               split=('train', 'valid', 'test'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = WikiText2(root=root, split='train')
        dataset_valid = WikiText2(root=root, split='valid')
        dataset_test = WikiText2(root=root, split='test')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_valid.get_dataset_size() == num_lines["valid"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.skip(reason="this ut has already tested")
    def test_squad2_by_register(self):
        """test squad2 by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('WikiText2',
                 root=root,
                 split=('train', 'valid', 'test')
                 )

class TestWikiText2Process(unittest.TestCase):
    r"""
    Test WikiText2 Process
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_wikitext2_process_no_vocab(self):
        r"""
        Test WikiText2 process with no vocab
        """

        test_dataset = WikiText2(
            root=os.path.join(os.path.expanduser('~'), ".mindnlp"),
            split="train"
        )

        test_dataset, vocab = WikiText2_Process(
            test_dataset, "text", text.BasicTokenizer())

        for i in test_dataset.create_tuple_iterator():
            assert i[0].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.skip(reason="this ut has already tested")
    def test_wikitext2_process_no_vocab_by_register(self):
        '''
        Test WikiText2 process with no vocab by register
        '''

        test_dataset = WikiText2(
            root=os.path.join(os.path.expanduser('~'), ".mindnlp"),
            split="train"
        )

        test_dataset, vocab = process('WikiText2', test_dataset, "text",
            text.BasicTokenizer())

        for i in test_dataset.create_tuple_iterator():
            assert i[0].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
