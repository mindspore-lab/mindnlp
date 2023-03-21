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
"""Test GPTNeo"""
import unittest
import numpy as np
import mindspore

from mindspore import Tensor
from mindnlp.models.gpt_neo import gpt_neo
from mindnlp.models.gpt_neo import gpt_neo_config


class TestModelingGPTNeo(unittest.TestCase):
    """
    Test GPTNeo
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_gptneo_self_attention(self):
        """
        Test GPTNeo Self Attention.
        """
        config = gpt_neo_config.GPTNeoConfig()
        model = gpt_neo.GPTNeoSelfAttention(
            config=config, attention_type=config.attention_types)

        hidden_states = Tensor(np.random.randint(
            0, 10, (2, 512, 2048)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (2, 512, 2048)

    def test_gptneo_attention(self):
        """
        Test GPTNeo Attention.
        """
        config = gpt_neo_config.GPTNeoConfig()
        model = gpt_neo.GPTNeoAttention(config=config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (2, 512, 2048)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (2, 512, 2048)

    def test_gptneo_mlp(self):
        """
        Test GPTNeo MLP.
        """
        intermediate_size = 8192
        config = gpt_neo_config.GPTNeoConfig()
        model = gpt_neo.GPTNeoMLP(
            intermediate_size=intermediate_size, config=config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (2, 512, 2048)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)

        assert attn_output.shape == (2, 512, 2048)
