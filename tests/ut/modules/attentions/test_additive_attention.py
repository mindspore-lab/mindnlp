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
"""Test Additive Attention"""

import numpy as np

import mindspore

from mindspore import ops
from mindspore import Tensor
from mindspore import context

from mindnlp.modules import AdditiveAttention
from ....common import MindNLPTestCase

class TestAdditiveAttention(MindNLPTestCase):
    r"""
    Test module Additive Attention
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_additive_attention_pynative(self):
        """
        unit test for additive attention with pynative mode.
        """
        context.set_context(mode=context.PYNATIVE_MODE)
        standard_normal = ops.StandardNormal(seed=114514)
        query = standard_normal((2, 32, 512))
        key = standard_normal((2, 20, 512))
        value = standard_normal((2, 20, 512))
        mask_shape = (2, 32, 20)
        mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        net = AdditiveAttention(hidden_dims=512)
        output, attn = net(query, key, value, mask=mask)

        assert output.shape == (2, 32, 512)
        assert attn.shape == (2, 32, 20)
