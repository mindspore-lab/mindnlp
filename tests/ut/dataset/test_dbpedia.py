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
Test DBpedia
"""
import os
import unittest
import pytest
import mindspore as ms
from mindnlp.dataset import DBpedia, DBpedia_Process
from mindnlp.dataset import load, process
from mindnlp.dataset.transforms import BasicTokenizer


class TestDBpedia(unittest.TestCase):
    r"""
    Test DBpedia
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_dbpedia(self):
        """Test DBpedia"""
        num_lines = {
            "train": 560000,
            "test": 70000,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train, dataset_test = DBpedia(
            root=root, split=("train", "test")
        )
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = DBpedia(root=root, split="train")
        dataset_test = DBpedia(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.skip(reason="this ut has already tested")
    def test_dbpedia_by_register(self):
        """test dbpedia by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load(
            "dbpedia",
            root=root,
            split=("train", "test"),
        )

class TestDBpediaProcess(unittest.TestCase):
    r"""
    Test DBpedia_Process
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_dbpedia_process(self):
        r"""
        Test DBpedia_Process
        """

        train_dataset, _ = DBpedia()
        train_dataset, vocab = DBpedia_Process(train_dataset)

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break

    @pytest.mark.skip(reason="this ut has already tested")
    def test_dbpedia_process_by_register(self):
        """test DBpedia process by register"""
        train_dataset, _ = DBpedia()
        train_dataset, vocab = process('DBpedia',
                                dataset=train_dataset,
                                column="title_text",
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )

        train_dataset = train_dataset.create_tuple_iterator()
        assert (next(train_dataset)[1]).dtype == ms.int32

        for _, value in vocab.vocab().items():
            assert isinstance(value, int)
            break
