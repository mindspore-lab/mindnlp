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
from mindnlp.dataset import IWSLT2017
from mindnlp.dataset import load


class TestIWSLT2017(unittest.TestCase):
    r"""
    Test IWSLT2017
    """

    def setUp(self):
        self.input = None

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

    def test_iwslt2017_by_register(self):
        """test iwslt2017 by register"""
        root = os.path.join(os.path.expanduser('~'), ".mindnlp")
        _ = load('iwslt2017',
                 root=root,
                 split=('train', 'valid', 'test'),
                 language_pair=('de', 'en')
                 )
