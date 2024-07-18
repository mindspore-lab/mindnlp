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
# pylint:disable=R0904
"""Test LUKE"""
import gc
import os
import unittest

import mindspore
import numpy as np
from mindspore import Tensor

from mindnlp.transformers.models.luke import luke_config, luke
from .....common import MindNLPTestCase


class TestModelingLUKE(MindNLPTestCase):
    r"""
    Test LUKE
    """

    def setUp(self):
        """
        Set up.
        """
        self.config = luke_config.LukeConfig(vocab_size=1000, num_hidden_layers=2)

    def test_luke_embeddings(self):
        r"""
        Test LukeEmbeddings
        """
        model = luke.LukeEmbeddings(self.config)
        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, (1, 128)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs.shape == (1, 128, 128)

    def test_luke_entity_embeddings(self):
        r"""
        Test LukeEntityEmbeddings
        """
        model = luke.LukeEntityEmbeddings(self.config)
        entity_ids = Tensor(np.random.randint(0, self.config.entity_vocab_size, (1,)), mindspore.int32)
        position_ids = Tensor(np.random.randn(1, 2), mindspore.int32)
        outputs = model(entity_ids, position_ids)
        assert outputs.shape == (1, 128)

    def test_luke_self_attention(self):
        r"""
        Test LukeSelfAttention
        """
        model = luke.LukeSelfAttention(self.config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_self_output(self):
        r"""
        Test LukeSelfOutput
        """
        model = luke.LukeSelfOutput(self.config)
        hidden_states = Tensor(np.random.randn(2, 128), mindspore.float32)
        input_tensor = Tensor(np.random.randn(2, 128), mindspore.float32)
        outputs = model(hidden_states, input_tensor)
        assert outputs.shape == (2, 128)

    def test_luke_attention(self):
        r"""
        Test LukeAttention
        """
        model = luke.LukeAttention(self.config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_intermediate(self):
        r"""
        Test LukeIntermediate
        """
        model = luke.LukeIntermediate(self.config)
        hidden_states = Tensor(np.random.randn(1, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (1, 3072)

    def test_luke_output(self):
        r"""
        Test LukeOutput
        """
        model = luke.LukeOutput(self.config)
        hidden_states = Tensor(np.random.randn(1, 3072), mindspore.float32)
        input_tensor = Tensor(np.random.rand(2, 128), mindspore.float32)
        output = model(hidden_states, input_tensor)
        assert output.shape == (2, 128)

    def test_luke_layer(self):
        r"""
        Test LukeLayer
        """
        model = luke.LukeLayer(self.config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_encoder(self):
        r"""
        test_LukeEncoder
        """
        model = luke.LukeEncoder(self.config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states, return_dict=False)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_pooler(self):
        r"""
        Test LukePooler
        """
        model = luke.LukePooler(self.config)
        hidden_states = Tensor(np.random.randn(128, 128, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (128, 128)

    def test_entity_prediction_head_transform(self):
        r"""
        Test EntityPredictionHeadTransform
        """
        model = luke.EntityPredictionHeadTransform(self.config)
        hidden_states = Tensor(np.random.randn(2, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 256)

    def test_entity_prediction_head(self):
        r"""
        Test EntityPredictionHead
        """
        model = luke.EntityPredictionHead(self.config)
        hidden_states = Tensor(np.random.randn(2, 128), dtype=mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 500)

    def test_luke_model(self):
        r"""
        Test LukeModel
        """
        model = luke.LukeModel(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 4)), dtype=mindspore.int32)
        outputs = model(input_ids, return_dict=False)
        assert outputs[0].shape == (2, 4, 128)

    def test_luke_lm_head(self):
        r"""
        Test LukeLMHead
        """
        model = luke.LukeLMHead(self.config)
        features = Tensor(np.random.randn(2, 128), mindspore.float32)
        output = model(features)
        assert output.shape == (2, self.config.vocab_size)

    def test_luke_for_masked_lm(self):
        r"""
        Test LukeForMaskedLM
        """
        model = luke.LukeForMaskedLM(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 4)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (2, 4, self.config.vocab_size)

    def test_luke_for_entity_classification(self):
        r"""
        Test LukeForEntityClassification
        """
        model = luke.LukeForEntityClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 128)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_entity_pair_classification(self):
        r"""
        Test LukeForEntityPairClassification
        """
        model = luke.LukeForEntityPairClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 128)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_entity_span_classification(self):
        r"""
        Test LukeForEntitySpanClassification
        """
        model = luke.LukeForEntitySpanClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 5, (2, 5)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 5, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 5, (2, 2)), mindspore.int32)
        entity_start_positions = Tensor(np.random.randint(0, 5, (2, 2)), mindspore.int32)
        entity_end_positions = Tensor(np.random.randint(0, 5, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids,
                        entity_start_positions=entity_start_positions, entity_end_positions=entity_end_positions)
        assert outputs[0].shape == (2, 2, 2)

    def test_luke_for_sequence_classification(self):
        r"""
        Test LukeForSequenceClassification
        """
        model = luke.LukeForSequenceClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 5)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_token_classification(self):
        r"""
        Test LukeForTokenClassification
        """
        model = luke.LukeForTokenClassification(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 5)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 5, 2)

    def test_luke_for_question_answering(self):
        r"""
        Test LukeForQuestionAnswering
        """
        model = luke.LukeForQuestionAnswering(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_multiple_choice(self):
        r"""
        Test LukeForMultipleChoice
        """
        model = luke.LukeForMultipleChoice(self.config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (1, 2)

    def tearDown(self) -> None:
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("~/.mindnlp"):
            os.removedirs("~/.mindnlp")
