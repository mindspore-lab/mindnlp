# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test Bloom"""

import gc
import os
import unittest
import pytest
import numpy as np

from mindspore import Tensor

from mindnlp.models.bloom import BloomConfig, BloomModel


class TestModelingBloom(unittest.TestCase):
    r"""
    Test Bloom
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = BloomConfig(vocab_size=1000,
                                 hidden_size=128,
                                 n_layer=2,
                                 n_head=8,
                                 intermediate_size=256)

    def test_bloom_model(self):
        r"""
        Test Bloom Model
        """
        model = BloomModel(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (2, 512)))

        hidden_states, _ = model(input_ids)
        assert hidden_states.shape == (2, 512, self.config.hidden_size)

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = BloomModel.from_pretrained('bigscience/bloom-560m', from_pt=True)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
