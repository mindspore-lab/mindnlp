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
import gc
import os
import unittest
import numpy as np
import mindspore

from mindspore import Tensor
from mindnlp.transformers.models.gpt_neo import gpt_neo, gpt_neo_config
from .....common import MindNLPTestCase


class TestModelingGPTNeo(MindNLPTestCase):
    """
    Test GPTNeo
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = gpt_neo_config.GPTNeoConfig(
            vocab_size=100,
            hidden_size=128,
            num_layers=4,
            attention_types=[[["global", "local"], 2]]
            )

    def test_gptneo_self_attention(self):
        """
        Test GPTNeo Self Attention.
        """
        model = gpt_neo.GPTNeoSelfAttention(
            config=self.config, attention_type=self.config.attention_types)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (1, 128, 128)

    def test_gptneo_attention(self):
        """
        Test GPTNeo Attention.
        """
        model = gpt_neo.GPTNeoAttention(config=self.config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        attn_output = model(hidden_states=hidden_states)[0]

        assert attn_output.shape == (1, 128, 128)

    def test_gptneo_mlp(self):
        """
        Test GPTNeo MLP.
        """
        intermediate_size = 512
        model = gpt_neo.GPTNeoMLP(
            intermediate_size=intermediate_size, config=self.config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        outputs = model(hidden_states=hidden_states)

        assert outputs.shape == (1, 128, 128)

    def test_gptneo_block(self):
        """
        Test GPTNeo Block.
        """
        model = gpt_neo.GPTNeoBlock(
            layer_id=0, config=self.config)

        hidden_states = Tensor(np.random.randint(
            0, 10, (1, 128, 128)), mindspore.float32)

        outputs = model(hidden_states=hidden_states)[0]

        assert outputs.shape == (1, 128, 128)

    def test_gptneo_model(self):
        """
        Test GPTNeo Model.
        """
        model = gpt_neo.GPTNeoModel(config=self.config)

        input_ids = Tensor(np.random.randint(
            0, 10, (1, 128)))

        outputs = model(input_ids)

        assert outputs[0].shape == (1, 128, 128)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                assert outputs[1][i][j].shape == (1, 16, 128, 8)

    def test_gptneo_for_causal_lm(self):
        """
        Test GPTNeo For CausalLM.
        """
        model = gpt_neo.GPTNeoForCausalLM(config=self.config)

        input_ids = Tensor(np.random.randint(
            0, 10, (1, 128)))

        outputs = model(input_ids)

        assert outputs[0].shape == (1, 128, 100)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                assert outputs[1][i][j].shape == (1, 16, 128, 8)

    def test_gptneo_for_sequence_classification(self):
        """
        Test GPTNeo For Sequence Classification.
        """
        model = gpt_neo.GPTNeoForSequenceClassification(config=self.config)

        input_ids = Tensor(np.random.randint(
            0, 10, (1, 128)))

        outputs = model(input_ids)

        assert outputs[0].shape == (1, 2)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                assert outputs[1][i][j].shape == (1, 16, 128, 8)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
