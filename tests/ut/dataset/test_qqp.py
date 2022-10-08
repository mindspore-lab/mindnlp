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
from mindnlp.dataset import QQP
from mindnlp.dataset import load


class TestQQP(unittest.TestCase):
    r"""
    Test QQP
    """

    def setUp(self):
        self.input = None

    def test_qqp(self):
        """Test qqp"""
        num_lines = {
            "train": 404290,
        }
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        dataset_train = QQP(root=root)
        assert dataset_train.get_dataset_size() == num_lines["train"]

    def test_qqp_by_register(self):
        """test qqp by register"""
        root = os.path.join(os.path.expanduser("~"), ".mindnlp")
        _ = load("QQP", root=root)
