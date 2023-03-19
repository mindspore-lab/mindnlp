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
"""Test GPT"""

import unittest
import numpy as np
import mindspore
from mindspore import Tensor
from mindnlp.models.gpt import gpt_config, gpt


class TestModelingGPT(unittest.TestCase):
    r"""
    Test GPT
    """
    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_gpt_mlp(self):
        r"""
        Test GPT MLP
        """
        intermediate_size = 3072
        config = gpt_config.GPTConfig()
        model = gpt.GPTMLP(intermediate_size, config)
        hidden_states = Tensor(np.random.randn(2, 512, 768), mindspore.float32)
        mlp_output = model(hidden_states)
        assert mlp_output.shape == (2, 512, 768)

    def test_gpt_attention(self):
        r"""
        Test GPT Attention
        """
        config = gpt_config.GPTConfig()
        model = gpt.GPTAttention(config)
        hidden_states = Tensor(np.random.randn(2, 512, 768), mindspore.float32)
        attn_output = model(hidden_states)
        assert attn_output[0].shape == (2, 512, 768)

    def test_gpt_block(self):
        r"""
        Test GPT Block
        """
        config = gpt_config.GPTConfig()
        model = gpt.GPTBlock(config)
        hidden_states = Tensor(np.random.randn(2, 512, 768), mindspore.float32)
        block_outputs = model(hidden_states)
        assert block_outputs[0].shape == (2, 512, 768)

    def test_gpt_model(self):
        r"""
        Test GPT Model
        """
        config = gpt_config.GPTConfig()
        model = gpt.GPTModel(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))
        model_outputs = model(input_ids)
        assert model_outputs[0].shape == (2, 512, 768)
