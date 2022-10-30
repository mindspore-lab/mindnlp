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
Test QQP
"""
import os
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import QQP, QQP_Process
from mindnlp.dataset import load, process
from mindnlp.dataset.transforms import BasicTokenizer


class TestQQP(unittest.TestCase):
    r"""
    Test QQP
    """

    def setUp(self):
        self.input = None

    @pytest.mark.dataset
    def test_qqp(self):
        """Test qqp"""
        num_lines = {
            "train": 404290,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train = QQP(root=root)
        assert dataset_train.get_dataset_size() == num_lines["train"]

    @pytest.mark.dataset
    def test_qqp_by_register(self):
        """test qqp by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load("QQP", root=root)

class TestQQPProcess(unittest.TestCase):
    r"""
    Test QQP_Process
    """

    def setUp(self):
        self.input = None

    @pytest.mark.dataset
    def test_qqp_process(self):
        r"""
        Test QQP_Process
        """

        train_dataset = QQP()
        train_dataset, vocab = QQP_Process(train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32
        assert (next(train_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.dataset
    def test_qqp_process_by_register(self):
        """test qqp process by register"""
        train_dataset = QQP()
        train_dataset, vocab = process('QQP',
                                dataset=train_dataset,
                                column=("question1", "question2"),
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32
        assert (next(train_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
