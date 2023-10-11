# pylint: disable=R0904
"""ut test for megatron_bert."""
import unittest
import numpy as np
import mindspore
from mindspore import Tensor
import mindnlp.models.megatron_bert


class TestModelingMegatronBert(unittest.TestCase):
    """
    test megatronBert
    """

    def setUp(self):
        """
        Set up
        """
        self.input = None
        self.config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

    def test_megatron_bert_embeddings(self):
        """
        test megatron_bert embedding layer
        """

        model = mindnlp.models.megatron_bert.MegatronBertEmbeddings(self.config)
        embedding_input = Tensor(np.random.randint(0, self.config.vocab_size, (2, 128)), mindspore.int32)
        attn_output, _ = model(embedding_input)
        assert attn_output.shape == (128, 128)

    def test_megatron_bert_self_attention(self):
        """
        test megatron_bert self-attention layer
        """
        model = mindnlp.models.megatron_bert.MegatronBertSelfAttention(self.config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)

        output = model(hidden_states)
        assert output[0].shape == (4, 32, 128)

    def test_megatron_bert_self_output(self):
        """
        test megatron_bert self-output layer
        """
        model = mindnlp.models.megatron_bert.MegatronBertSelfOutput(self.config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)
        residual = Tensor(np.random.randn(4, 32, 128), mindspore.float32)

        output = model(hidden_states, residual)
        assert output[0].shape == (32, 128)

    def test_megatron_bert_attention(self):
        """
        test megatron_bert attention layer
        """
        model = mindnlp.models.megatron_bert.MegatronBertAttention(self.config)
        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (4, 32, 128)

    def test_megatron_bert_intermediate(self):
        """
        test megatron_bert intermediate layer
        """
        model = mindnlp.models.megatron_bert.MegatronBertIntermediate(self.config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (4, 32, 128)

    def test_megatron_bert_output(self):
        """
        test megatron_bert output
        """
        model = mindnlp.models.megatron_bert.MegatronBertOutput(self.config)

        hidden_states = Tensor(np.random.randn(2, 8, 128), mindspore.float32)
        input_tensor = Tensor(np.random.randn(2, 8, 128), mindspore.float32)
        output = model(hidden_states, input_tensor)
        assert output.shape == (2, 8, 128)

    def test_megatron_bert_layer(self):
        """
        test megatron_bert layer
        """
        model = mindnlp.models.megatron_bert.MegatronBertLayer(self.config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (2, 32, 128)

    def test_megatron_bert_encoder(self):
        """
        test megatron_bert encoding
        """
        model = mindnlp.models.megatron_bert.MegatronBertEncoder(self.config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (2, 32, 128)

    def test_megatron_bert_pooler(self):
        """
        test megatron_bert encoding
        """
        model = mindnlp.models.megatron_bert.MegatronBertPooler(self.config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (128,)

    def test_megatron_bert_prediction_head_transform(self):
        """
        test MegatronBertPredictionHeadTransform
        """
        model = mindnlp.models.megatron_bert.MegatronBertPredictionHeadTransform(self.config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 32, 128)

    def test_megatron_bert_lm_prediction_head(self):
        """
        test MegatronBertLMPredictionHead
        """
        model = mindnlp.models.megatron_bert.MegatronBertLMPredictionHead(self.config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 32, 100)

    def test_megatron_bert_only_mlm_head(self):
        """
        test MegatronBertOnlyMLMHead
        """
        model = mindnlp.models.megatron_bert.MegatronBertOnlyMLMHead(self.config)

        sequence_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(sequence_output)
        assert output.shape == (2, 32, 100)

    def test_megatron_bert_only_nsp_head(self):
        """
        test MegatronBertOnlyNSPHead
        """
        model = mindnlp.models.megatron_bert.MegatronBertOnlyNSPHead(self.config)

        pooled_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(pooled_output)
        assert output.shape == (2, 32, 2)

    def test_megatron_bert_pretraining_heads(self):
        """
        test MegatronBertPreTrainingHeads
        """
        model = mindnlp.models.megatron_bert.MegatronBertPreTrainingHeads(self.config)

        sequence_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        pooled_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(sequence_output, pooled_output)
        assert output[0].shape == (2, 32, 100)
        assert output[1].shape == (2, 32, 2)

    def test_megatron_bert_model(self):
        """
        test MegatronBertModel
        """
        model = mindnlp.models.megatron_bert.MegatronBertModel(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 32, 128)
        assert output[1].shape == (4, 128)

    def test_megatron_bert_for_pretraining(self):
        """
        test MegatronBertPreTraining
        """
        model = mindnlp.models.megatron_bert.MegatronBertForPreTraining(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 100)
        assert output[1].shape == (4, 2)

    def test_megatron_bert_for_causal_lm(self):
        """
        test MegatronBertForCausalLM
        """
        self.config.is_decoder = True
        model = mindnlp.models.megatron_bert.MegatronBertForCausalLM(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 16]), mindspore.int32)
        output = model(input_ids)

        assert output[0].shape == (4, 16, 100)
        assert len(output[1]) == 24
        assert output[1][0][0].shape == (4, 16, 16, 8)

    def test_megatron_bert_for_masked_lm(self):
        """
        test MegatronBertForMaskedLM
        """
        model = mindnlp.models.megatron_bert.MegatronBertForMaskedLM(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 100)

    def test_megatron_bert_for_next_sentence_prediction(self):
        """
        test MegatronBertForNextSentencePrediction
        """
        self.config.is_decoder = True
        model = mindnlp.models.megatron_bert.MegatronBertForNextSentencePrediction(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [2, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (2, 2)

    def test_megatron_bert_for_sequence_classification(self):
        """
        test MegatronBertForSequenceClassification
        """
        self.config.is_decoder = True
        model = mindnlp.models.megatron_bert.MegatronBertForSequenceClassification(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 2)

    def test_megatron_bert_for_multiple_choice(self):
        """
        test MegatronBertForMultipleChoice
        """
        model = mindnlp.models.megatron_bert.MegatronBertForMultipleChoice(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [1, 2, 4, 8]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 2)

    def test_megatron_bert_for_token_classification(self):
        """
        test MegatronBertForTokenClassification
        """
        model = mindnlp.models.megatron_bert.MegatronBertForTokenClassification(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 2)

    def test_megatron_bert_for_question_answering(self):
        """
        test MegatronBertForQuestionAnswering
        """
        model = mindnlp.models.megatron_bert.MegatronBertForQuestionAnswering(self.config)

        input_ids = Tensor(np.random.randint(0, self.config.vocab_size, [4, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 32)
        assert output[1].shape == (4, 32)
