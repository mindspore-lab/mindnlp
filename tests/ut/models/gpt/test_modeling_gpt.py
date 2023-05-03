# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Test GPT"""

import unittest
import pytest
import numpy as np
from ddt import ddt, data

import mindspore
from mindspore import Tensor

from mindnlp import ms_jit
from mindnlp.models.gpt import GPTConfig, GPTModel, MLP, Attention, Block, \
    GPTLMHeadModel, GPTDoubleHeadsModel, GPTForSequenceClassification

@ddt
class TestModelingGPT(unittest.TestCase):
    r"""
    Test GPT
    """
    def setUp(self):
        """
        Set up.
        """
        self.config = GPTConfig(n_layer=2)

    def test_gpt_mlp(self):
        r"""
        Test GPT MLP
        """
        intermediate_size = 3072
        model = MLP(intermediate_size, self.config)
        hidden_states = Tensor(np.random.randn(2, 512, 768), mindspore.float32)
        mlp_output = model(hidden_states)
        assert mlp_output.shape == (2, 512, 768)

    def test_gpt_attention(self):
        r"""
        Test GPT Attention
        """
        model = Attention(self.config.n_embd, self.config.n_positions, self.config)
        hidden_states = Tensor(np.random.randn(2, 512, 768), mindspore.float32)
        attn_output = model(hidden_states)
        assert attn_output[0].shape == (2, 512, 768)

    def test_gpt_block(self):
        r"""
        Test GPT Block
        """
        model = Block(self.config.n_positions, self.config)
        hidden_states = Tensor(np.random.randn(2, 512, 768), mindspore.float32)
        block_outputs = model(hidden_states)
        assert block_outputs[0].shape == (2, 512, 768)

    @data(True, False)
    def test_gpt_model(self, jit):
        r"""
        Test GPT Model
        """
        model = GPTModel(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        def forward(input_ids):
            model_outputs = model(input_ids)
            return model_outputs

        if jit:
            forward = ms_jit(forward)

        model_outputs = forward(input_ids)

        assert model_outputs[0].shape == (2, 512, 768)

    @data(True, False)
    def test_gpt_lmhead_model(self, jit):
        r"""
        Test GPT2 LMHead Model
        """
        model = GPTLMHeadModel(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        def forward(input_ids):
            model_outputs = model(input_ids)
            return model_outputs

        if jit:
            forward = ms_jit(forward)
        model_outputs = forward(input_ids)

        assert model_outputs[0].shape == (2, 512, 40478)

    @data(True, False)
    def test_gpt_double_heads_model(self, jit):
        r"""
        Test model GPT Model with pynative mode
        """
        model = GPTDoubleHeadsModel(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        def forward(input_ids):
            model_outputs = model(input_ids)
            return model_outputs

        if jit:
            forward = ms_jit(forward)
        model_outputs = forward(input_ids)

        assert model_outputs[0].shape == (2, 512, 40478)

    @data(True, False)
    def test_gpt_for_sequence_classification(self, jit):
        r"""
        Test GPT For Sequence Classification
        """
        model = GPTForSequenceClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (1, 512)))

        def forward(input_ids):
            model_outputs = model(input_ids)
            return model_outputs

        if jit:
            forward = ms_jit(forward)
        model_outputs = forward(input_ids)

        assert model_outputs[0].shape == (1, 2)


    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = GPTModel.from_pretrained('openai-gpt')

    @pytest.mark.download
    def test_gpt_double_heads_model_from_pretrained(self):
        """test from pretrained"""
        _ = GPTDoubleHeadsModel.from_pretrained('openai-gpt', from_pt=True)

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = GPTModel.from_pretrained('openai-gpt', from_pt=True)
