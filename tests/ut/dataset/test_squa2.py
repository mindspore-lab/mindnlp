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
Test SQuAD2
"""

import os
import unittest
from mindnlp.dataset import SQuAD2, load


class TestSQuAD2(unittest.TestCase):
    r"""
    Test SQuAD2
    """

    def setUp(self):
        self.input = None

    def test_squad2(self):
        """Test SQuAD2"""
        num_lines = {
            "train": 130319,
            "dev": 11873,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_dev = SQuAD2(root=root, split=('train', 'dev'))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

        dataset_train = SQuAD2(root=root, split='train')
        dataset_dev = SQuAD2(root=root, split='dev')
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_dev.get_dataset_size() == num_lines["dev"]

    def test_squad2_by_register(self):
        """test squad2 by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('squad2',
                 root=root,
                 split=('train', 'dev')
                 )
