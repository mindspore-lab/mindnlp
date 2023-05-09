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
"""Test GPT2"""
import unittest
import pytest
import numpy as np

import mindspore

from mindspore import Tensor

from mindnlp.models.gpt2 import config_gpt2, gpt2


class TestModelingGPT2(unittest.TestCase):
    r"""
    Test GPT2
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_gpt2_attention(self):
        r"""
        Test GPT2 Attention
        """
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2Attention(config)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 512, 768)), mindspore.float32)

        attn_output, _ = model(hidden_states)
        assert attn_output.shape == (2, 512, 768)

    def test_gpt2_mlp(self):
        r"""
        Test GPT2 MLP
        """
        intermediate_size = 3072
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2MLP(intermediate_size, config)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 512, 768)), mindspore.float32)

        hidden_states = model(hidden_states)
        assert hidden_states.shape == (2, 512, 768)

    def test_gpt2_block(self):
        r"""
        Test GPT2 Block
        """
        layer_idx = 0
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2Block(config, layer_idx)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 512, 768)), mindspore.float32)

        outputs = model(hidden_states)
        assert outputs[0].shape == (2, 512, 768)

    def test_gpt2_model(self):
        r"""
        Test GPT2 Model
        """
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2Model(config)

        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        hidden_states, presents = model(input_ids)
        assert hidden_states.shape == (2, 512, 768)
        assert presents[0][0].shape == (2, 12, 512, 64)
        assert presents[0][1].shape == (2, 12, 512, 64)

    def test_gpt2_lmhead_model(self):
        r"""
        Test GPT2 LMHead Model
        """
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2LMHeadModel(config)

        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        lm_logits, transformer_outputs = model(input_ids)
        assert lm_logits.shape == (2, 512, 50257)
        assert transformer_outputs[0][0].shape == (2, 12, 512, 64)
        assert transformer_outputs[0][1].shape == (2, 12, 512, 64)

    def test_gpt2_double_heads_model(self):
        r"""
        Test model GPT2 Model with pynative mode
        """
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2DoubleHeadsModel(config)

        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        logits, mc_logits, past_key_values = model(input_ids)

        assert logits.shape == (2, 512, 50257)
        assert mc_logits.shape == (2,)
        assert past_key_values[0][0].shape == (2, 12, 512, 64)
        assert past_key_values[0][1].shape == (2, 12, 512, 64)

    def test_gpt2_for_sequence_classification(self):
        r"""
        Test GPT2 For Sequence Classification
        """
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2ForSequenceClassification(config)

        input_ids = Tensor(np.random.randint(0, 10, (1, 512)))

        pooled_logits, transformer_outputs = model(input_ids)
        assert pooled_logits.shape == (1, 2)
        assert transformer_outputs[0][0].shape == (1, 12, 512, 64)
        assert transformer_outputs[0][1].shape == (1, 12, 512, 64)

    def test_gpt2_for_token_classification(self):
        r"""
        Test model GPT2 Model with pynative mode
        """
        config = config_gpt2.GPT2Config(n_layer=2)
        model = gpt2.GPT2ForTokenClassification(config)

        input_ids = Tensor(np.random.randint(0, 10, (2, 512)))

        logits = model(input_ids)
        assert logits[0].shape == (2, 512, 2)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = gpt2.GPT2Model.from_pretrained('gpt2')

    @pytest.mark.download
    def test_gpt2_lm_head_model_from_pretrained(self):
        """test from pretrained"""
        _ = gpt2.GPT2LMHeadModel.from_pretrained('gpt2', from_pt=True)

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = gpt2.GPT2Model.from_pretrained('gpt2', from_pt=True)
