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
Test AG_NEWS
"""
import os
import unittest
import mindspore as ms
from mindspore.dataset.text import BasicTokenizer
from mindnlp.dataset import AG_NEWS, AG_NEWS_Process
from mindnlp.dataset import load, process


class TestAGNEWS(unittest.TestCase):
    r"""
    Test AG_NEWS
    """

    def setUp(self):
        self.input = None

    def test_agnews(self):
        """Test agnews"""
        num_lines = {
            "train": 120000,
            "test": 7600,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_test = AG_NEWS(
            root=root, split=("train", "test"))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = AG_NEWS(root=root, split="train")
        dataset_test = AG_NEWS(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    def test_agnews_by_register(self):
        """test agnews by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('AG_NEWS', root=root, split=('train', 'test'),)


class TestAGNEWSProcess(unittest.TestCase):
    r"""
    Test AG_NEWS_Process
    """

    def setUp(self):
        self.input = None

    def test_agnews_process(self):
        r"""
        Test AG_NEWS_Process
        """

        train_dataset, _ = AG_NEWS()
        agnews_dataset = AG_NEWS_Process(train_dataset)
        agnews_dataset = agnews_dataset.create_tuple_iterator()

        assert (next(agnews_dataset)[1]).dtype == ms.int32

    def test_agnews_process_by_register(self):
        """test agnews process by register"""
        train_dataset, _ = AG_NEWS()
        train_dataset = process('AG_NEWS',
                                dataset=train_dataset,
                                column="text",
                                tokenizer=BasicTokenizer(),
                                vocab=None
                                )
