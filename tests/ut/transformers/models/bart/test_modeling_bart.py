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
import mindspore
import numpy as np
from mindspore import Tensor

import mindnlp
from mindnlp.transformers import BartConfig
from mindnlp.transformers.models.bart import bart
from ..model_test import ModelTest


class TestModelingBart(ModelTest):
    """
    Test Bart
    """

    def setUp(self):
        """
        Set up.
        """
        super().setUp()
        self.input_ids = np.array([
                [71, 82, 18, 33, 46, 91, 2],
                [68, 34, 26, 58, 30, 82, 2],
            ])
        self.config = BartConfig(vocab_size=1024,max_position_embeddings=128,d_model=128)

    def test_bart_learned_positional_embedding(self):
        """
        Test BartLearnedPositionalEmbedding
        """
        model = bart.BartLearnedPositionalEmbedding(self.config.max_position_embeddings,self.config.d_model)
        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, 10)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs.shape == (1, 10, self.config.max_position_embeddings)

    def test_bart_attention(self):
        """
        Test BartAttention
        """
        model = bart.BartAttention(
            embed_dim=self.config.d_model,
            num_heads=self.config.encoder_attention_heads,
            dropout=self.config.attention_dropout,
        )

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randn(1, 2, self.config.d_model), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs[0].shape == (1, 2, self.config.d_model)

    def test_bart_encoder_layer(self):
        """
        Test BartEncoderLayer
        """
        model = bart.BartEncoderLayer(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randn(1, 2, self.config.d_model), mindspore.float32)
        attention_mask = Tensor(np.random.randn(1, 1, 2, 2) > 0, mindspore.bool_)
        layer_head_mask = Tensor(np.random.randn(16) > 0, mindspore.bool_)
        outputs = model(hidden_states,attention_mask,layer_head_mask)
        assert outputs[0].shape == (1, 2, self.config.d_model)

    def test_bart_decoder_layer(self):
        """
        Test BartDecoderLayer
        """
        model = bart.BartDecoderLayer(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randn(1, 2, self.config.d_model), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs[0].shape == (1, 2, self.config.d_model)

    def test_bart_classification_head(self):
        """
        Test BartClassificationHead
        """
        model = bart.BartClassificationHead(
            self.config.d_model,
            self.config.d_model,
            self.config.num_labels,
            self.config.classifier_dropout,
        )

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randn(1, 2, self.config.d_model), mindspore.float32)
        outputs = model(hidden_states)
        assert outputs.shape == (1, 2, 3)

    def test_bart_encoder(self):
        """
        Test BartEncoder
        """
        model = bart.BartEncoder(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, 2)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 2, self.config.d_model)

    def test_bart_decoder(self):
        """
        Test BartDecoder
        """
        model = bart.BartDecoder(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, 2)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 2, self.config.d_model)

    def test_bart_model(self):
        """
        Test BartModel
        """
        model = bart.BartModel(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, 2)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 2, self.config.d_model)

    def test_bart_for_conditional_generation(self):
        """
        Test BartForConditionalGeneration
        """
        model = bart.BartForConditionalGeneration(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, 2)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (1, 2, self.config.vocab_size)

    def test_bart_for_sequence_classification(self):
        """
        Test BartForSequenceClassification
        """
        model = bart.BartForSequenceClassification(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(self.input_ids, mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (2, 3)

    def test_bart_for_question_answering(self):
        """
        Test BartForQuestionAnswering
        """
        model = bart.BartForQuestionAnswering(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(self.input_ids, mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (2, 7)

    def test_bart_for_causal_lm(self):
        """
        Test BartForCausalLM
        """
        model = bart.BartForCausalLM(self.config)

        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(self.input_ids, mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (2, 7, self.config.vocab_size)
