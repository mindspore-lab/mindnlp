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
# pylint:disable=W0212
"""Test T5"""
import unittest
import pytest
import numpy as np

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor

from mindnlp.models.t5 import (T5Config,
                               T5LayerNorm,
                               T5DenseActDense,
                               T5DenseGatedActDense,
                               T5LayerFF,
                               T5Attention,
                               T5LayerSelfAttention,
                               T5LayerCrossAttention,
                               T5Block,
                               T5PreTrainedModel,
                               T5Stack,
                               T5Model,
                               T5ForConditionalGeneration,
                               T5EncoderModel)
class TestModelingT5(unittest.TestCase):
    r"""
    Test T5
    """
    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_t5_layer_norm(self):
        r"""
        Test T5LayerNorm
        """
        hidden_size = 512
        model = T5LayerNorm((hidden_size,), eps = 1e-6)

        input_ids = Tensor(np.random.randn(hidden_size), mindspore.float32)

        outputs = model(input_ids)
        assert outputs.shape == (hidden_size, )

    def test_t5_dense_act_dense(self):
        r"""
        Test T5DenseActDense
        """
        config = T5Config()
        model = T5DenseActDense(config)

        input_ids = Tensor(np.random.randn(2, config.d_model), mindspore.float32)

        outputs = model(input_ids)
        assert outputs.shape == (2, config.d_model)

    def test_t5_dense_gated_act_dense(self):
        r"""
        Test T5DenseGatedActDense
        """
        config = T5Config()
        model = T5DenseGatedActDense(config)

        input_ids = Tensor(np.random.randn(2, config.d_model), mindspore.float32)

        outputs = model(input_ids)
        assert outputs.shape == (2, config.d_model)

    def test_t5_layer_ff(self):
        r"""
        Test T5LayerFF
        """
        config = T5Config()
        model = T5LayerFF(config)

        input_ids = Tensor(np.random.randn(2, config.d_model), mindspore.float32)

        outputs = model(input_ids)
        assert outputs.shape == (2, config.d_model)

    def test_t5_attention(self):
        r"""
        Test T5Attention
        """
        config = T5Config()
        model = T5Attention(config)

        input_ids = Tensor(np.random.randn(4, 64, 512), mindspore.float32)

        outputs = model(input_ids)
        assert outputs[0].shape == (4, 64, 512)
        assert outputs[2].shape == (1, 8, 64, 64)

    def test_t5_layer_self_attention(self):
        r"""
        Test T5LayerSelfAttention
        """
        config = T5Config()
        model = T5LayerSelfAttention(config)

        input_ids = Tensor(np.random.randn(4, 64, 512), mindspore.float32)

        outputs = model(input_ids)
        assert outputs[0].shape == (4, 64, 512)
        assert outputs[2].shape == (1, 8, 64, 64)

    def test_t5_layer_cross_attention(self):
        r"""
        Test T5LayerCrossAttention
        """
        config = T5Config()
        model = T5LayerCrossAttention(config)

        input_ids = Tensor(np.random.randn(4, 64, 512), mindspore.float32)

        outputs = model(input_ids, key_value_states=None)
        assert outputs[0].shape == (4, 64, 512)
        assert outputs[2].shape == (1, 8, 64, 64)

    def test_t5_block(self):
        r"""
        Test T5Block
        """
        config = T5Config()
        model = T5Block(config)

        input_ids = Tensor(np.random.randn(4, 64, 512), mindspore.float32)

        outputs = model(input_ids)
        assert outputs[0].shape == (4, 64, 512)
        assert outputs[1].shape == (1, 8, 64, 64)

    def test_t5_pretrainedmodel(self):
        r"""
        Test T5PreTrainedModel._shift_right
        """
        decoder_start_token_id = 0
        config = T5Config(decoder_start_token_id = decoder_start_token_id)
        model = T5PreTrainedModel(config)
        input_ids = Tensor([[1, 2, 3, -100, -100, -100], [1, 2, 3, -100, -100, -100]])

        outputs = model._shift_right(input_ids)
        assert ops.all(outputs == Tensor([[0, 1, 2, 3, 0, 0], [0, 1, 2, 3, 0, 0]]))

    def test_t5_stack(self):
        r"""
        Test T5Stack
        """
        config = T5Config(dropout_rate=0, return_dict = False)
        embed = nn.Embedding(1024, 512)
        model = T5Stack(config, embed_tokens=embed)
        input_ids = Tensor(np.random.randint(0, 100, (1, 4)), dtype=mindspore.int64)

        outputs = model(input_ids, use_cache=False)
        assert outputs[0].shape == (1, 4, 512)

    def test_t5_model(self):
        r"""
        Test T5Model
        """
        config = T5Config(decoder_start_token_id = 0,dropout_rate=0, return_dict = False, num_layers=2)
        model = T5Model(config)
        input_ids = Tensor(np.random.randint(0,100,(1,10)), dtype=mindspore.int64)
        decoder_input_ids = Tensor(np.random.randint(0,100,(1,20)), dtype=mindspore.int64)
        outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        assert outputs[0].shape == (1, 20, 512)
        for i in range(len(outputs[1])):
            for j in range(len(outputs[1][i])):
                if j in [0,1]:
                    assert outputs[1][i][j].shape == (1, 8, 20, 64)
                else:
                    assert outputs[1][i][j].shape == (1, 8, 10, 64)
        assert outputs[2].shape == (1, 10, 512)

    def test_t5_forconditionalgeneration(self):
        r"""
        Test T5ForConditionalGeneration
        """
        config = T5Config(decoder_start_token_id = 0,dropout_rate=0, return_dict = False, num_layers=2)
        model = T5ForConditionalGeneration(config)
        input_ids = Tensor(np.random.randint(0,100,(1,10)), dtype=mindspore.int64)
        labels = Tensor(np.random.randint(0,100,(1,20)), dtype=mindspore.int32)
        outputs = model(input_ids=input_ids, labels = labels)
        assert outputs[1].shape == (1, 20, 32128)
        for i in range(len(outputs[2])):
            for j in range(len(outputs[2][i])):
                if j in [0,1]:
                    assert outputs[2][i][j].shape == (1, 8, 20, 64)
                else:
                    assert outputs[2][i][j].shape == (1, 8, 10, 64)
        assert outputs[3].shape == (1, 10, 512)

    def test_t5_encodermodel(self):
        r"""
        Test T5EncoderModel
        """
        config = T5Config(decoder_start_token_id = 0,dropout_rate=0, return_dict = False, num_layers=2)
        model = T5EncoderModel(config)
        input_ids = Tensor(np.random.randint(0,100,(1,10)), dtype=mindspore.int64)

        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 10, 512)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = T5Model.from_pretrained('t5-small')

    @pytest.mark.download
    def test_from_pretrained_from_pt(self):
        """test from pt"""
        _ = T5Model.from_pretrained('t5-small', from_pt=True)
