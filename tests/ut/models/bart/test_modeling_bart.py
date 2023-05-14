# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# ============================================================================
"""Test Bart"""
import unittest

import mindspore
import numpy as np
from mindspore import Tensor

from mindnlp.models.bart import BartConfig
from mindnlp.models.bart import bart


class TestModelingBart(unittest.TestCase):
    """
    Test Bart
    """

    def setUp(self):
        """
        Set up.
        """
        self.input_ids = np.array([
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
                [5, 97, 17, 39, 94, 40, 2],
                [76, 83, 94, 25, 70, 78, 2],
                [87, 59, 41, 35, 48, 66, 2],
                [55, 13, 16, 58, 5, 2, 1],  # note padding
                [64, 27, 31, 51, 12, 75, 2],
                [52, 64, 86, 17, 83, 39, 2],
                [48, 61, 9, 24, 71, 82, 2],
                [26, 1, 60, 48, 22, 13, 2],
                [21, 5, 62, 28, 14, 76, 2],
                [45, 98, 37, 86, 59, 48, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
                [70, 70, 50, 9, 28, 0, 2],
            ])

    def test_bart_LearnedPositionalEmbedding(self):
        """
        Test BartLearnedPositionalEmbedding
        """
        config = BartConfig()
        model = bart.BartLearnedPositionalEmbedding(config.max_position_embeddings,config.d_model)
        input_ids = Tensor(np.random.randn(1, 10), mindspore.int32)
        outputs = model(input_ids)
        assert outputs.shape == (1, 10, 1024)

    def test_bart_attention(self):
        """
        Test BartAttention
        """
        config = BartConfig()
        model = bart.BartAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        hidden_states = Tensor(np.random.randn(1, 2, 1024), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs[0].shape == (1, 2, 1024)

    def test_bart_EncoderLayer(self):
        r"""
        Test BartEncoderLayer
        """
        config = BartConfig()
        model = bart.BartEncoderLayer(config)
        hidden_states = Tensor(np.random.randn(1, 2, 1024), mindspore.float32)
        attention_mask = Tensor(np.random.randn(1, 1, 2, 2) > 0, mindspore.bool_)
        layer_head_mask = Tensor(np.random.randn(16) > 0, mindspore.bool_)
        outputs = model(hidden_states,attention_mask,layer_head_mask)
        assert outputs[0].shape == (1, 2, 1024)

    def test_bart_DecoderLayer(self):
        r"""
        Test BartDecoderLayer
        """
        config = BartConfig()
        model = bart.BartDecoderLayer(config)
        hidden_states = Tensor(np.random.randn(1, 2, 1024), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs[0].shape == (1, 2, 1024)

    def test_bart_ClassificationHead(self):
        r"""
        Test BartClassificationHead
        """
        config = BartConfig()
        model = bart.BartClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )
        hidden_states = Tensor(np.random.randn(1, 2, 1024), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs.shape == (1, 2, 3)

    def test_bart_Encoder(self):
        r"""
        Test BartEncoder
        """
        config = BartConfig()
        model = bart.BartEncoder(config)
        input_ids = Tensor(np.random.randn(1, 10), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 10, 1024)

    def test_bart_decoder(self):
        r"""
        Test BartDecoder
        """
        config = BartConfig()
        model = bart.BartDecoder(config)
        input_ids = Tensor(np.random.randn(1, 10), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 10, 1024)

    def test_bart_model(self):
        r"""
        Test BartModel
        """
        config = BartConfig()
        model = bart.BartModel(config)
        input_ids = Tensor(np.random.randn(1, 10), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 10, 1024)

    def test_bart_ForConditionalGeneration(self):
        r"""
        Test BartForConditionalGeneration
        """
        config = BartConfig()
        model = bart.BartForConditionalGeneration(config)
        input_ids = Tensor(np.random.randn(1, 10), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 10, 50265)

    def test_bart_ForSequenceClassification(self):
        r"""
        Test BartForSequenceClassification
        """
        config = BartConfig()
        model = bart.BartForSequenceClassification(config)
        input_ids = Tensor(self.input_ids, mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (24, 3)

    def test_bart_ForQuestionAnswering(self):
        r"""
        Test BartForQuestionAnswering
        """
        config = BartConfig()
        model = bart.BartForQuestionAnswering(config)
        input_ids = Tensor(self.input_ids, mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (24, 7)

    def test_bart_ForCausalLM(self):
        r"""
        Test BartForCausalLM
        """
        config = BartConfig()
        model = bart.BartForCausalLM(config)
        input_ids = Tensor(self.input_ids, mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (24, 7, 50265)