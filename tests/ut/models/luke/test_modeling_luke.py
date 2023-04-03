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
import unittest

import mindspore
import numpy as np
from mindspore import Tensor

from mindnlp.models.luke import luke_config, luke


class TestModelingLUKE(unittest.TestCase):
    r"""
    Test LUKE
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_luke_embeddings(self):
        r"""
        Test LukeEmbeddings
        """
        config = luke_config.LukeConfig()
        model = luke.LukeEmbeddings(config)
        input_ids = Tensor(np.random.randn(1, 2), mindspore.int32)
        outputs = model(input_ids)
        assert outputs.shape == (1, 2, 128)

    def test_luke_entity_embeddings(self):
        r"""
        Test LukeEntityEmbeddings
        """
        config = luke_config.LukeConfig()
        model = luke.LukeEmbeddings(config)
        entity_ids = Tensor(np.random.randn(1, ), mindspore.int32)
        position_ids = Tensor(np.random.randn(1, 2), mindspore.int32)
        outputs = model(entity_ids, position_ids)
        assert outputs.shape == (1, 2, 128)

    def test_luke_self_attention(self):
        r"""
        Test LukeSelfAttention
        """
        config = luke_config.LukeConfig()
        model = luke.LukeSelfAttention(config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_self_output(self):
        r"""
        Test LukeSelfOutput
        """
        config = luke_config.LukeConfig()
        model = luke.LukeSelfOutput(config)
        hidden_states = Tensor(np.random.randn(2, 128), mindspore.float32)
        input_tensor = Tensor(np.random.randn(2, 128), mindspore.float32)
        outputs = model(hidden_states, input_tensor)
        assert outputs.shape == (2, 128)

    def test_luke_attention(self):
        r"""
        Test LukeAttention
        """
        config = luke_config.LukeConfig()
        model = luke.LukeAttention(config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_intermediate(self):
        r"""
        Test LukeIntermediate
        """
        config = luke_config.LukeConfig()
        model = luke.LukeIntermediate(config)
        hidden_states = Tensor(np.random.randn(1, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (1, 3072)

    def test_luke_output(self):
        r"""
        Test LukeOutput
        """
        config = luke_config.LukeConfig()
        model = luke.LukeOutput(config)
        hidden_states = Tensor(np.random.randn(1, 3072), mindspore.float32)
        input_tensor = Tensor(np.random.rand(2, 128), mindspore.float32)
        output = model(hidden_states, input_tensor)
        assert output.shape == (2, 128)

    def test_luke_layer(self):
        r"""
        Test LukeLayer
        """
        config = luke_config.LukeConfig()
        model = luke.LukeLayer(config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_encoder(self):
        r"""
        test_LukeEncoder
        """
        config = luke_config.LukeConfig()
        model = luke.LukeEncoder(config)
        word_hidden_states = Tensor(np.random.randn(1, 2, 128), mindspore.float32)
        entity_hidden_states = Tensor(np.random.randn(1, 4, 128), mindspore.float32)
        outputs = model(word_hidden_states, entity_hidden_states, return_dict=False)
        assert outputs[0].shape == (1, 2, 128)
        assert outputs[1].shape == (1, 4, 128)

    def test_luke_pooler(self):
        r"""
        Test LukePooler
        """
        config = luke_config.LukeConfig()
        model = luke.LukePooler(config)
        hidden_states = Tensor(np.random.randn(128, 128, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (128, 128)

    def test_entity_prediction_head_transform(self):
        r"""
        Test EntityPredictionHeadTransform
        """
        config = luke_config.LukeConfig()
        model = luke.EntityPredictionHeadTransform(config)
        hidden_states = Tensor(np.random.randn(2, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 256)

    def test_entity_prediction_head(self):
        r"""
        Test EntityPredictionHead
        """
        config = luke_config.LukeConfig()
        model = luke.EntityPredictionHead(config)
        hidden_states = Tensor(np.random.randn(2, 128), dtype=mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 500)

    def test_luke_model(self):
        r"""
        Test LukeModel
        """
        config = luke_config.LukeConfig()
        model = luke.LukeModel(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 4)), dtype=mindspore.int32)
        outputs = model(input_ids, return_dict=False)
        assert outputs[0].shape == (2, 4, 128)

    def test_luke_lm_head(self):
        r"""
        Test LukeLMHead
        """
        config = luke_config.LukeConfig()
        model = luke.LukeLMHead(config)
        features = Tensor(np.random.randn(2, 128), mindspore.float32)
        output = model(features)
        assert output.shape == (2, 100)

    def test_luke_for_masked_lm(self):
        r"""
        Test LukeForMaskedLM
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForMaskedLM(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 4)), mindspore.int32)
        outputs = model(input_ids)
        assert outputs[0].shape == (2, 4, 100)

    def test_luke_for_entity_classification(self):
        r"""
        Test LukeForEntityClassification
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForEntityClassification(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 128)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_entity_pair_classification(self):
        r"""
        Test LukeForEntityPairClassification
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForEntityPairClassification(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 128)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_entity_span_classification(self):
        r"""
        Test LukeForEntitySpanClassification
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForEntitySpanClassification(config)
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
        config = luke_config.LukeConfig()
        model = luke.LukeForSequenceClassification(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 5)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_token_classification(self):
        r"""
        Test LukeForTokenClassification
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForTokenClassification(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 5)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 5, 2)

    def test_luke_for_question_answering(self):
        r"""
        Test LukeForQuestionAnswering
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForQuestionAnswering(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (2, 2)

    def test_luke_for_multiple_choice(self):
        r"""
        Test LukeForMultipleChoice
        """
        config = luke_config.LukeConfig()
        model = luke.LukeForMultipleChoice(config)
        input_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        entity_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        position_ids = Tensor(np.random.randint(0, 10, (2, 2)), mindspore.int32)
        outputs = model(input_ids=input_ids, entity_ids=entity_ids, entity_position_ids=position_ids)
        assert outputs[0].shape == (1, 2)
