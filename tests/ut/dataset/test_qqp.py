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
import shutil
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import QQP, QQP_Process
from mindnlp import load_dataset, process

from mindnlp.transforms import BasicTokenizer


class TestQQP(unittest.TestCase):
    r"""
    Test QQP
    """

    @classmethod
    def setUpClass(cls):
        cls.root = os.path.join(os.path.expanduser("~"), ".mindnlp")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    @pytest.mark.download
    @pytest.mark.local
    def test_qqp(self):
        """Test qqp"""
        num_lines = {
            "train": 404290,
        }
        dataset_train = QQP(root=self.root)
        assert dataset_train.get_dataset_size() == num_lines["train"]

    @pytest.mark.download
    @pytest.mark.local
    def test_qqp_by_register(self):
        """test qqp by register"""
        _ = load_dataset("QQP", root=self.root)

    @pytest.mark.download
    @pytest.mark.local
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

    @pytest.mark.download
    @pytest.mark.local
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
