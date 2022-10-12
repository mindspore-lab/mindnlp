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
Test IWSLT2017
"""
import os
import unittest
import pytest
import mindspore
from mindspore.dataset import text
from mindnlp.dataset import IWSLT2017, IWSLT2017_Process
from mindnlp.dataset import load, process


class TestIWSLT2017(unittest.TestCase):
    r"""
    Test IWSLT2017
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_iwslt2017(self):
        """Test IWSLT2017"""
        num_lines = {
            "train": 206112,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, _, _ = IWSLT2017(root=root,
                                        split=(
                                            'train', 'valid', 'test'),
                                        language_pair=(
                                            'de', 'en')
                                        )
        assert dataset_train.get_dataset_size() == num_lines["train"]

        dataset_train = IWSLT2017(
            root=root, split='train', language_pair=('de', 'en'))
        assert dataset_train.get_dataset_size() == num_lines["train"]

    @pytest.mark.skip(reason="this ut has already tested")
    def test_iwslt2017_by_register(self):
        """test iwslt2017 by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('iwslt2017',
                 root=root,
                 split=('train', 'valid', 'test'),
                 language_pair=('de', 'en')
                 )


class TestIWSLT2017Process(unittest.TestCase):
    r"""
    Test IWSLT2017 Process
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_iwslt2017_process_no_vocab(self):
        r"""
        Test IWSLT2017 process with no vocab
        """

        test_dataset = IWSLT2017(
            root=os.path.join(os.path.expanduser('~'), ".mindnlp"),
            split="test",
            language_pair=("de", "en"),
        )

        test_dataset, vocab = IWSLT2017_Process(
            test_dataset, text.BasicTokenizer(), "translation")

        for i in test_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.skip(reason="this ut has already tested")
    def test_iwslt2017_process_no_vocab_by_register(self):
        '''
        Test IWSLT2017 process with no vocab by register
        '''

        test_dataset = IWSLT2017(
            root=os.path.join(os.path.expanduser('~'), ".mindnlp"),
            split="test",
            language_pair=("de", "en")
        )

        test_dataset, vocab = process('IWSLT2017', test_dataset,
            text.BasicTokenizer(), "translation")

        for i in test_dataset.create_tuple_iterator():
            assert i[1].dtype == mindspore.int32
            break

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
