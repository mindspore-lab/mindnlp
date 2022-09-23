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
"""Test Location Aware Attention"""

import unittest
import numpy as np

import mindspore

from mindspore import ops, Tensor
from mindspore import context

from mindnlp.modules.attentions import LocationAwareAttention


class TestLocationAwareAttention(unittest.TestCase):
    r"""
    Test module Location Aware Attention
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_location_aware_attention_pynative(self):
        """
        unit test for location aware attention with pynative mode.
        """
        context.set_context(mode=context.PYNATIVE_MODE)
        batch_size, seq_len, enc_d, dec_d, attn_d = 2, 40, 32, 20, 512
        standard_normal = ops.StandardNormal(seed=114514)
        query = standard_normal((batch_size, 1, dec_d))
        value = standard_normal((batch_size, seq_len, enc_d))
        last_attn = standard_normal((batch_size, seq_len))
        net = LocationAwareAttention(
            decoder_dim=dec_d,
            encoder_dim=enc_d,
            attn_dim=attn_d,
            smoothing=False)
        mask_shape = (batch_size, seq_len)
        mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        net.set_mask(mask)
        cont, attn = net(query, value, last_attn)

        assert cont.shape == (batch_size, 1, enc_d)
        assert attn.shape == (batch_size, seq_len)

    def test_location_aware_attention_graph(self):
        """
        unit test for location aware attention whit graph mode.
        """
        context.set_context(mode=context.GRAPH_MODE)
        batch_size, seq_len, enc_d, dec_d, attn_d = 2, 40, 32, 20, 512
        standard_normal = ops.StandardNormal(seed=114514)
        query = standard_normal((batch_size, 1, dec_d))
        value = standard_normal((batch_size, seq_len, enc_d))
        last_attn = standard_normal((batch_size, seq_len))
        net = LocationAwareAttention(
            decoder_dim=dec_d,
            encoder_dim=enc_d,
            attn_dim=attn_d,
            smoothing=False)
        mask_shape = (batch_size, seq_len)
        mask = Tensor(np.ones(mask_shape), mindspore.bool_)
        net.set_mask(mask)
        cont, attn = net(query, value, last_attn)

        assert cont.shape == (batch_size, 1, enc_d)
        assert attn.shape == (batch_size, seq_len)
