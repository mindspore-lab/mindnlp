# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Test Linear Attention"""

import unittest
import numpy as np

import mindspore

from mindspore import ops
from mindspore import Tensor
from mindspore import context

from mindnlp.modules import LinearAttention
from ....common import MindNLPTestCase

class TestLinearAttention(MindNLPTestCase):
    r"""
    Test module Linear Attention
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_linear_attention_pynative(self):
        """
        unit test for linear attention with pynative mode.
        """
        context.set_context(mode=context.PYNATIVE_MODE)
        standard_normal = ops.StandardNormal(seed=114514)
        query = standard_normal((2, 32, 512))
        key = standard_normal((2, 20, 512))
        value = standard_normal((2, 20, 500))
        net = LinearAttention(query_dim=32, key_dim=20, hidden_dim=512)
        mask_shape = (2, 32, 20)
        mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        output, attn = net(query, key, value, mask)

        assert output.shape == (2, 32, 500)
        assert attn.shape == (2, 32, 20)
