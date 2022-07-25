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
"""Test Dot Attention"""

import unittest
import numpy as np

import mindspore
from mindspore import Tensor

from text.modules.attentions import DotAttention


class TestDotAttention(unittest.TestCase):
    r"""
    Test module DotAttention
    """

    def setUp(self):
        self.input = None

    def test_dot_attention(self):
        net = DotAttention(dropout=0.9)
        q = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        k = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        v = Tensor(np.ones((2, 1024, 512)), mindspore.float32)
        output = net(q, k, v)

        assert output.shape == (2, 1024, 512)
