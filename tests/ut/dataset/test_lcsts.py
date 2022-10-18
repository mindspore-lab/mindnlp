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
Test LCSTS
"""

import os
import unittest
import pytest
import mindspore
from mindspore.dataset import text
from mindnlp.dataset import LCSTS, LCSTS_Process, load, process


class TestLCSTS(unittest.TestCase):
    r"""
    Test LCSTS
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_lcsts(self):
        """Test LCSTS"""
        num_lines = {
            "train": 1470769,
            "dev": 10666,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_dev = LCSTS(root=root, split=('train', 'dev'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

        dataset_train = LCSTS(root=root, split='train')
        dataset_dev = LCSTS(root=root, split='dev')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

    @pytest.mark.skip(reason="this ut has already tested")
    def test_lcsts_by_register(self):
        """test lcsts by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('lcsts',
                 root=root,
                 split=('train', 'dev')
                 )

class TestLCSTSProcess(unittest.TestCase):
    r"""
    Test LCSTS Process
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_lcsts_process_no_vocab(self):
        r"""
        Test LCSTS process with no vocab
        """

        test_dataset = LCSTS(
            root=os.path.join(os.path.expanduser('~'), ".mindnlp"),
            split="train"
        )

        test_dataset, vocab = LCSTS_Process(
            test_dataset, "target", text.BasicTokenizer())

        for i in test_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.skip(reason="this ut has already tested")
    def test_lcsts_process_no_vocab_by_register(self):
        '''
        Test LCSTS process with no vocab by register
        '''

        test_dataset = LCSTS(
            root=os.path.join(os.path.expanduser('~'), ".mindnlp"),
            split="train"
        )

        test_dataset, vocab = process('LCSTS', test_dataset, "target",
            text.BasicTokenizer())

        for i in test_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
