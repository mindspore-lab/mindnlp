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
"""Test Llama"""
import gc
import unittest
import numpy as np
import mindspore

from mindspore import Tensor
from mindnlp.models.llama.llama_hf import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)
from mindnlp.models.llama.llama_hf_config import LlamaConfig


class TestModelingLlama(unittest.TestCase):
    """
    Test Llama
    """
    def setUp(self):
        """
        Set up.
        """
        self.config = LlamaConfig(num_hidden_layers=2, num_attention_heads=8, hidden_size=128)


    def test_llama_rms_norm(self):
        r"""
        test_llama_rms_norm
        """
        hidden_size = 512
        model = LlamaRMSNorm((hidden_size,), eps = 1e-6)

        input_ids = Tensor(np.random.randn(hidden_size), mindspore.float32)

        outputs = model(input_ids)
        assert outputs.shape == (hidden_size, )

    def test_llama_mlp_norm(self):
        r"""
        test_llama_mlp_norm
        """
        batch_size = 2
        seq_len = 128
        hidden_size = 128
        model = LlamaMLP(hidden_size, hidden_size * 4, "silu")

        input_ids = Tensor(np.random.randint(0, 10, (batch_size, seq_len, hidden_size)), mindspore.float32)
        outputs = model(input_ids)
        assert outputs.shape == (batch_size, seq_len, hidden_size)

    def test_llama_attention(self):
        r"""
        Test Llama attention
        """
        model = LlamaAttention(self.config)

        input_ids = Tensor(np.random.randn(2, self.config.max_position_embeddings, 128), mindspore.float32)
        position_ids = Tensor([np.arange(0, self.config.max_position_embeddings)
                               , np.arange(0, self.config.max_position_embeddings)], mindspore.int64)
        outputs = model(input_ids, position_ids=position_ids)
        assert outputs[0].shape == (2, self.config.max_position_embeddings, 128)

    def test_llama_decoderlayer(self):
        r"""
        test_llama_decoderlayer
        """
        model = LlamaDecoderLayer(self.config)

        input_ids = Tensor(np.random.randn(2, self.config.max_position_embeddings, 128), mindspore.float32)
        position_ids = Tensor([np.arange(0, self.config.max_position_embeddings)
                               , np.arange(0, self.config.max_position_embeddings)], mindspore.int64)
        outputs = model(input_ids, position_ids=position_ids)
        assert outputs[0].shape == (2, self.config.max_position_embeddings, 128)

    def test_llama_model(self):
        """
        Test Llama Model.
        """
        model = LlamaModel(self.config)

        input_ids = Tensor(np.random.randint(
            0, 100, (2, 128)))

        outputs = model(input_ids)

        assert outputs[0].shape == (2, 128, 128)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                assert outputs[1][i][j].shape == (2, 8, 128, 16)

    def test_llama_for_causal_lm(self):
        """
        test_llama_for_causal_lm
        """
        model = LlamaForCausalLM(self.config)

        input_ids = Tensor(np.random.randint(
            0, 100, (2, 128)))

        outputs = model(input_ids)

        assert outputs[0].shape == (2, 128, 32000)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                assert outputs[1][i][j].shape == (2, 8, 128, 16)

    def test_llama_for_sequence_classification(self):
        """
        test_llama_for_sequence_classification
        """
        model = LlamaForSequenceClassification(self.config)

        input_ids = Tensor(np.random.randint(
            0, 10, (2, 128)))

        outputs = model(input_ids)

        assert outputs[0].shape == (2, 2)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                assert outputs[1][i][j].shape == (2, 8, 128, 16)

    def tearDown(self) -> None:
        gc.collect()
