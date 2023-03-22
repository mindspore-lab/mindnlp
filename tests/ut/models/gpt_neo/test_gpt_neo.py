# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        config = gpt_neo_config.GPTNeoConfig(vocab_size=100, hidden_size=128)
        model = gpt_neo.GPTNeoSelfAttention(
            config=config, attention_type=config.attention_types)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (1, 128, 128)

    def test_gptneo_attention(self):
        """
        Test GPTNeo Attention.
        """
        config = gpt_neo_config.GPTNeoConfig(vocab_size=100, hidden_size=128)
        model = gpt_neo.GPTNeoAttention(config=config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (1, 128, 128)

    def test_gptneo_mlp(self):
        """
        Test GPTNeo MLP.
        """
        intermediate_size = 512
        config = gpt_neo_config.GPTNeoConfig(vocab_size=100, hidden_size=128)
        model = gpt_neo.GPTNeoMLP(
            intermediate_size=intermediate_size, config=config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)

        assert attn_output.shape == (1, 128, 128)

    def test_gptneo_block(self):
        """
        Test GPTNeo Block.
        """
        config = gpt_neo_config.GPTNeoConfig(vocab_size=100, hidden_size=128)
        model = gpt_neo.GPTNeoBlock(
            layer_id=0, config=config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (1, 128, 128)

    def test_gptneo_model(self):
        """
        Test GPTNeo Model.
        """
        config = gpt_neo_config.GPTNeoConfig(vocab_size=100, hidden_size=64)
        model = gpt_neo.GPTNeoModel(config=config)

        input_ids = Tensor(np.random.randint(
            0, 10, (1, 128, 64)))

        attn_output = model(input_ids)

        assert attn_output[0].shape == (1, 128, 64, 64)
        for i in range(len(attn_output[1])):
            for j in range(len(attn_output[1][i])):
                assert attn_output[1][i][j].shape == (128, 16, 64, 4)
