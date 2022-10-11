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
Test MRPC
"""
import os
import unittest
import pytest
from mindnlp.dataset import MRPC
from mindnlp.dataset import load


class TestMRPC(unittest.TestCase):
    r"""
    Test MRPC
    """

    def setUp(self):
        self.input = None

    @pytest.mark.skip(reason="this ut has already tested")
    def test_mrpc(self):
        """Test mrpc"""
        num_lines = {
            "train": 4076,
            "test": 1725,
        }
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        dataset_train, dataset_test = MRPC(
            root=root, split=("train", "test"))
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

        dataset_train = MRPC(root=root, split="train")
        dataset_test = MRPC(root=root, split="test")
        assert dataset_train.get_dataset_size() == num_lines["train"]
        assert dataset_test.get_dataset_size() == num_lines["test"]

    @pytest.mark.skip(reason="this ut has already tested")
    def test_mrpc_by_register(self):
        """test mrpc by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('MRPC', root=root, split=('train', 'test'),)
