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
"""Test Self Attention"""

import unittest
import numpy as np

import mindspore

from mindspore import ops
from mindspore import context
from mindspore import Tensor

from mindnlp.modules import SelfAttention
from ....common import MindNLPTestCase

class TestSelfAttention(MindNLPTestCase):
    r"""
    Test module Self Attention
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_self_attention_pynative(self):
        """
        unit test for self attention with pynative mode.
        """
        context.set_context(mode=context.PYNATIVE_MODE)
        standard_normal = ops.StandardNormal(seed=114514)
        query = standard_normal((2, 32, 512))
        key = standard_normal((2, 20, 512))
        value = standard_normal((2, 20, 512))
        mask_shape = (2, 32, 20)
        mask = Tensor(np.ones(mask_shape), mindspore.bool_)

        # use dot-product attention default dot-product
        net = SelfAttention()
        output, attn = net(query, key, value, mask)
        assert output.shape == (2, 32, 512)
        assert attn.shape == (2, 32, 20)

        # use cosine attention
        net = SelfAttention(attention_mode="cosine")
        output, attn = net(query, key, value, mask)
        assert output.shape == (2, 32, 512)
        assert attn.shape == (2, 32, 20)

        # use additive attention
        net = SelfAttention(attention_mode="add")
        output, attn = net(query, key, value, mask)
        assert output.shape == (2, 32, 512)
        assert attn.shape == (2, 32, 20)
