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
# pylint:disable=R0904
"""
Test TinyBert
"""
import pytest
import numpy as np
import mindspore
from mindspore import ops

from mindnlp.transformers.models.tinybert.tinybert import TinyBertConfig, TinyBertEmbeddings, TinyBertSelfAttention, \
    TinyBertAttention, TinyBertSelfOutput, TinyBertIntermediate, TinyBertOutput, \
    TinyBertLayer, TinyBertEncoder, TinyBertPooler, TinyBertPredictionHeadTransform, \
    TinyBertLMPredictionHead, TinyBertOnlyMLMHead, TinyBertOnlyNSPHead, TinyBertPreTrainingHeads, \
    TinyBertModel, TinyBertForPreTraining, TinyBertFitForPreTraining, TinyBertForNextSentencePrediction, \
    TinyBertForMaskedLM, TinyBertForSentencePairClassification, TinyBertForSequenceClassification

from .....common import MindNLPTestCase

class TestTinyBert(MindNLPTestCase):
    """
    Test TinyBert Models
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up config
        """

        cls.bert_config = TinyBertConfig(
            vocab_size=200,
            hidden_size=128,
            intermediate_size=256,
            num_attention_heads=8,
            num_hidden_layers=2)

    def test_tiny_bert_embedding(self):
        """
        Test BertEmbeddings
        """

        bert_embeddings = TinyBertEmbeddings(self.bert_config)
        input_ids = mindspore.Tensor(np.random.randint(0, self.bert_config.vocab_size, (2, 128)))
        output = bert_embeddings(input_ids)
        assert output.shape == (2, 128, self.bert_config.hidden_size)

    def test_tiny_bert_self_attention(self):
        """
        Test TinyBertSelfAttention
        """

        bert_self_attention = TinyBertSelfAttention(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        input_mask = ops.ones((2, 1, 1, 3), dtype=mindspore.float32)
        self_output, layer_att = bert_self_attention(input_tensor, input_mask)
        assert self_output.shape == (2, 3, 128)
        assert layer_att.shape == (2, 8, 3, 3)

    def test_tiny_bert_attention(self):
        """
        Test TinyBertAttention
        """

        bert_attention = TinyBertAttention(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        input_mask = ops.ones((2, 1, 1, 3), dtype=mindspore.float32)
        attention_output, layer_att = bert_attention(input_tensor, input_mask)
        assert attention_output.shape == (2, 3, 128)
        assert layer_att.shape == (2, 8, 3, 3)

    def test_tiny_bert_self_output(self):
        """
        Test TinyBertSelfOutput
        """

        bert_self_output = TinyBertSelfOutput(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        attention_output = bert_self_output(input_tensor, input_tensor)
        assert attention_output.shape == (2, 3, 128)

    def test_tiny_bert_intermediate(self):
        """
        Test TinyBertIntermediate
        """

        bert_intermediate = TinyBertIntermediate(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        intermediate_output = bert_intermediate(input_tensor)
        assert intermediate_output.shape == (2, 3, 256)

    def test_tiny_bert_output(self):
        """
        Test TinyBertOutput
        """

        bert_output = TinyBertOutput(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        intermediate_output = mindspore.Tensor(
            np.random.rand(2, 3, 256), dtype=mindspore.float32)
        layer_output = bert_output(intermediate_output, input_tensor)
        assert layer_output.shape == (2, 3, 128)

    def test_tiny_bert_layer(self):
        """
        Test TinyBertLayer
        """

        bert_layer = TinyBertLayer(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        input_mask = ops.ones((2, 1, 1, 3), dtype=mindspore.float32)
        hidden_states, layer_att = bert_layer(input_tensor, input_mask)
        assert hidden_states.shape == (2, 3, 128)
        assert layer_att.shape == (2, 8, 3, 3)

    def test_tiny_bert_encoder(self):
        """
        Test TinyBertEncoder
        """

        bert_encoder = TinyBertEncoder(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        input_mask = ops.ones((2, 1, 1, 3), dtype=mindspore.float32)
        encoded_layers, layer_atts = bert_encoder(input_tensor, input_mask)
        assert encoded_layers[0].shape == (2, 3, 128)
        assert layer_atts[0].shape == (2, 8, 3, 3)

    def test_tiny_bert_pooler(self):
        """
        Test TinyBertPooler
        """

        bert_pooler = TinyBertPooler(self.bert_config)
        input_encoded_layers = []
        for _ in range(13):
            input_encoded_layers.append(mindspore.Tensor(
                np.random.rand(2, 3, 128), dtype=mindspore.float32))

        pooled_output = bert_pooler(input_encoded_layers)
        assert pooled_output.shape == (2, 128)

    def test_tiny_bert_prediction_head_transform(self):
        """
        Test TinyBertPredictionHeadTransform
        """

        bert_prediction_head_transform = TinyBertPredictionHeadTransform(
            self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        hidden_states = bert_prediction_head_transform(input_tensor)
        assert hidden_states.shape == (2, 3, 128)

    def test_tiny_bert_lm_prediction_head(self):
        """
        Test TinyBertLMPredictionHead
        """

        bert_embeddings = TinyBertEmbeddings(self.bert_config)
        bert_lm_prediction_head = TinyBertLMPredictionHead(
            self.bert_config, bert_embeddings.word_embeddings.weight)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        prediction_scores = bert_lm_prediction_head(input_tensor)
        assert prediction_scores.shape == (2, 3, 200)

    def test_tiny_bert_only_mlm_head(self):
        """
        Test TinyBertOnlyMLMHead
        """

        bert_embeddings = TinyBertEmbeddings(self.bert_config)
        bert_only_mlm_head = TinyBertOnlyMLMHead(
            self.bert_config, bert_embeddings.word_embeddings.weight)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        prediction_scores = bert_only_mlm_head(input_tensor)
        assert prediction_scores.shape == (2, 3, 200)

    def test_tiny_bert_only_nsp_head(self):
        """
        Test TinyBertOnlyNSPHead
        """

        bert_only_nsp_head = TinyBertOnlyNSPHead(self.bert_config)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 128), dtype=mindspore.float32)
        seq_relationship_score = bert_only_nsp_head(input_tensor)
        assert seq_relationship_score.shape == (2, 2)

    def test_tiny_bert_pretraining_heads(self):
        """
        Test TinyBertPreTrainingHeads
        """

        bert_embeddings = TinyBertEmbeddings(self.bert_config)
        bert_pretraining_heads = TinyBertPreTrainingHeads(
            self.bert_config, bert_embeddings.word_embeddings.weight)
        input_tensor = mindspore.Tensor(
            np.random.rand(2, 3, 128), dtype=mindspore.float32)
        pooled_output = mindspore.Tensor(
            np.random.rand(2, 128), dtype=mindspore.float32)
        prediction_scores, seq_relationship_score = bert_pretraining_heads(
            input_tensor, pooled_output)
        assert prediction_scores.shape == (2, 3, 200)
        assert seq_relationship_score.shape == (2, 2)

    def test_tiny_bert_model(self):
        """
        Test TinyBertModel
        """

        bert_model = TinyBertModel(self.bert_config)
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)

        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)

        sequence_output, pooled_output = bert_model(
            ms_input_ids, ms_token_type_ids, ms_input_mask, output_all_encoded_layers=False, output_att=False)
        assert sequence_output.shape == (2, 3, 128)
        assert pooled_output.shape == (2, 128)

    def test_tiny_bert_for_pretraining(self):
        """
        TinyBertForPreTraining
        """

        bert_for_pretraining = TinyBertForPreTraining(
            self.bert_config)
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)

        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)

        prediction_scores, seq_relationship_score = bert_for_pretraining(
            ms_input_ids, ms_token_type_ids, ms_input_mask)
        assert prediction_scores.shape == (2, 3, 200)
        assert seq_relationship_score.shape == (2, 2)

    def test_tiny_bert_fit_for_pretraining(self):
        """
        Test TinyBertFitForPreTraining
        """

        bert_fit_for_pretraining = TinyBertFitForPreTraining(
            self.bert_config)
        # original
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
        # ms
        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)
        # forward
        masked_lm_logits_scores, seq_relationship_score = bert_fit_for_pretraining(
            ms_input_ids, ms_token_type_ids, ms_input_mask)
        assert masked_lm_logits_scores[0].shape == (2, 8, 3, 3)
        assert seq_relationship_score[0].shape == (2, 3, 768)

    def test_tiny_bert_for_masked_lm(self):
        """
        Test TinyBertForMaskedLM
        """

        bert_for_masked_lm = TinyBertForMaskedLM(self.bert_config)
        # original
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
        # ms
        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)
        ms_masked_lm_logits_scores = bert_for_masked_lm(
            ms_input_ids, ms_token_type_ids, ms_input_mask)
        assert ms_masked_lm_logits_scores.shape == (2, 3, 200)

    def test_tiny_bert_for_next_sentence_prediction(self):
        """
        Test TinyBertForNextSentencePrediction
        """

        bert_for_next_sentence_prediction = TinyBertForNextSentencePrediction(
            self.bert_config)
        # original
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
        # ms
        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)
        seq_relationship_logits = bert_for_next_sentence_prediction(
            ms_input_ids, ms_token_type_ids, ms_input_mask)
        assert seq_relationship_logits.shape == (2, 2)

    def test_tiny_bert_for_sentence_pair_classification(self):
        """
        TinyBertForSentencePairClassification
        """

        bert_for_sentence_pair_classification = TinyBertForSentencePairClassification(
            self.bert_config, 2)
        # original
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
        # ms
        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)
        logits = bert_for_sentence_pair_classification(
            ms_input_ids, ms_input_ids, ms_token_type_ids, ms_token_type_ids, ms_input_mask, ms_input_mask)
        assert logits.shape == (2, 2)

    def test_tiny_bert_for_sequence_classification(self):
        """
        Test TinyBertForSequenceClassification
        """

        bert_for_sequence_classification = TinyBertForSequenceClassification(
            self.bert_config, 2)
        # original
        input_ids = np.array([[31, 51, 99], [15, 5, 0]], dtype=np.int64)
        input_mask = np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int64)
        token_type_ids = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
        # ms
        ms_input_ids = mindspore.Tensor(input_ids, dtype=mindspore.int64)
        ms_input_mask = mindspore.Tensor(input_mask, dtype=mindspore.int64)
        ms_token_type_ids = mindspore.Tensor(
            token_type_ids, dtype=mindspore.int64)
        logits, att_output, sequence_output = bert_for_sequence_classification(
            ms_input_ids, ms_token_type_ids, ms_input_mask)
        assert logits.shape == (2, 2)
        assert att_output[0].shape == (2, 8, 3, 3)
        assert sequence_output[0].shape == (2, 3, 128)

    @pytest.mark.download
    def test_from_pretrained(self):
        """test from pretrained"""
        _ = TinyBertModel.from_pretrained('tinybert_6L_zh')

    @pytest.mark.download
    def test_from_pretrained_path(self):
        """test from pretrained"""
        _ = TinyBertModel.from_pretrained('.mindnlp/models/tinybert_6L_zh')
