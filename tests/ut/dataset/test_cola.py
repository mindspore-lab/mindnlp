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
Test CoLA
"""
import os
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import CoLA, CoLA_Process
from mindnlp.dataset import load, process
from mindnlp.dataset.transforms import BasicTokenizer


class TestCoLA(unittest.TestCase):
    r"""
    Test CoLA
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_cola(self):
        """Test cola"""
        num_lines = {
            "train": 8551,
            "dev": 527,
            "test": 516,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train, dataset_dev, dataset_test = CoLA(
            root=root, split=("train", "dev", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = CoLA(root=root, split="train")
        dataset_dev = CoLA(root=root, split="dev")
        dataset_test = CoLA(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.skip(reason="this ut has already tested")
    def test_cola_by_register(self):
        """test cola by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load(
            "CoLA",
            root=root,
            split=("train", "dev", "test"),
        )

class TestCoLAProcess(unittest.TestCase):
    r"""
    Test CoLA_Process
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_cola_process(self):
        r"""
        Test CoLA_Process
        """

        train_dataset, _, _ = CoLA()
        train_dataset, vocab = CoLA_Process(train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.skip(reason="this ut has already tested")
    def test_cola_process_by_register(self):
        """test cola process by register"""
        train_dataset, _, _ = CoLA()
        train_dataset, vocab = process('CoLA',
                                dataset=train_dataset,
                                column="sentence",
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[2]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
