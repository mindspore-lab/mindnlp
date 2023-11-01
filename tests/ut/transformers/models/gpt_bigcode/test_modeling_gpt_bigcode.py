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
"""Test GPTBigCode"""
import gc
import os
import unittest
import pytest
import numpy as np

import mindspore

from mindspore import Tensor

from mindnlp.transformers.models.gpt_bigcode import gpt_bigcode_config, gpt_bigcode, gpt_bigcode_tokenizer


class TestModelingGPTBigCode(unittest.TestCase):
    r"""
    Test GPTBigCode
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = gpt_bigcode_config.GPTBigCodeConfig(
            n_layer=2, n_embd=128, n_head=8, n_inner=256, pad_token_id=0)

    def test_gpt_bigcode_attention(self):
        r"""
        Test GPTBigCode Attention
        """
        model = gpt_bigcode.GPTBigCodeAttention(self.config)

        hidden_states = Tensor(np.random.randint(
            0, self.config.vocab_size, (2, 512, self.config.n_embd)), mindspore.float32)

        attn_output, _ = model(hidden_states)
        assert attn_output.shape == (2, 512, self.config.n_embd)

    def test_gpt_bigcode_mlp(self):
        r"""
        Test GPTBigCode MLP
        """
        model = gpt_bigcode.GPTBigCodeMLP(self.config.n_inner, self.config)

        hidden_states = Tensor(np.random.randint(
            0, self.config.vocab_size, (2, 512, self.config.n_embd)), mindspore.float32)

        hidden_states = model(hidden_states)
        assert hidden_states.shape == (2, 512, self.config.n_embd)

    def test_gpt_bigcode_block(self):
        r"""
        Test GPTBigCode Block
        """
        layer_idx = 0
        model = gpt_bigcode.GPTBigCodeBlock(self.config, layer_idx)

        hidden_states = Tensor(np.random.randint(
            0, self.config.vocab_size, (2, 512, self.config.n_embd)), mindspore.float32)

        outputs = model(hidden_states)
        assert outputs[0].shape == (2, 512, self.config.n_embd)

    def test_gpt_bigcode_model(self):
        r"""
        Test GPTBigCode Model
        """
        model = gpt_bigcode.GPTBigCodeModel(self.config)

        input_ids = Tensor(np.random.randint(
            0, self.config.vocab_size, (2, 512)))

        output = model(input_ids)
        assert output["last_hidden_state"].shape == (
            2, 512, self.config.n_embd)
        assert output["past_key_values"][0].shape == (
            2, 512, self.config.n_embd // (self.config.n_head // 2))
        assert output["past_key_values"][1].shape == (
            2, 512, self.config.n_embd // (self.config.n_head // 2))

    def test_gpt_bigcode_for_sequence_classification(self):
        r"""
        Test GPTBigCode For Sequence Classification
        """
        model = gpt_bigcode.GPTBigCodeForSequenceClassification(self.config)

        input_ids = Tensor(np.random.randint(
            0, self.config.vocab_size, (2, 512)))

        output = model(input_ids)
        assert output["logits"].shape == (2, 2, 2)
        assert output["past_key_values"][0].shape == (
            2, 512, self.config.n_embd // (self.config.n_head // 2))

    def test_gpt_bigcode_for_token_classification(self):
        r"""
        Test model GPTBigCode Model with pynative mode
        """
        model = gpt_bigcode.GPTBigCodeForTokenClassification(self.config)

        input_ids = Tensor(np.random.randint(
            0, self.config.vocab_size, (2, 512)))

        logits = model(input_ids)
        assert logits[0].shape == (2, 512, 2)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        model = gpt_bigcode.GPTBigCodeModel.from_pretrained(
            'gpt_bigcode-santacoder')

        tokenizer = gpt_bigcode_tokenizer.GPTBigCodeTokenizer.from_pretrained(
            "gpt_bigcode-santacoder")

        inputs = tokenizer("Hello, my dog is cute")

        _ = model(**inputs)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
