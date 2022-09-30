# Copyright 2022 Huawei Technologies Co., LtdSQuAD1
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
Test SQuAD1
"""

import os
import unittest
from mindnlp.dataset import SQuAD1, load


class TestSQuAD1(unittest.TestCase):
    r"""
    Test SQuAD1
    """

    def setUp(self):
        self.input = None

    def test_squad1(self):
        """Test SQuAD1"""
        num_lines = {
            "train": 87599,
            "dev": 10570,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_dev = SQuAD1(root=root, split=('train', 'dev'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

        dataset_train = SQuAD1(root=root, split='train')
        dataset_dev = SQuAD1(root=root, split='dev')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

    def test_squad1_by_register(self):
        """test squad1 by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('squad1',
                 root=root,
                 split=('train', 'dev')
                 )
