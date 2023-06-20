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
        # self.config = MegatronBertConfig()

    def test_megatron_bert_embeddings(self):
        """
        test megatron_bert embedding layer
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertEmbeddings(config)
        embedding_input = Tensor(np.random.randint(0, 1000, (2, 128)), mindspore.int32)
        attn_output, _ = model(embedding_input)
        assert attn_output.shape == (128, 128)

    def test_megatron_bert_self_attention(self):
        """
        test megatron_bert self-attention layer
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertSelfAttention(config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)

        output = model(hidden_states)
        assert output[0].shape == (4, 32, 128)

    def test_megatron_bert_self_output(self):
        """
        test megatron_bert self-output layer
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertSelfOutput(config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)
        residual = Tensor(np.random.randn(4, 32, 128), mindspore.float32)

        output = model(hidden_states, residual)
        assert output[0].shape == (32, 128)

    def test_megatron_bert_attention(self):
        """
        test megatron_bert attention layer
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertAttention(config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (4, 32, 128)

    def test_megatron_bert_intermediate(self):
        """
        test megatron_bert intermediate layer
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertIntermediate(config)

        hidden_states = Tensor(np.random.randn(4, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (4, 32, 128)

    def test_megatron_bert_output(self):
        """
        test megatron_bert output
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertOutput(config)

        hidden_states = Tensor(np.random.randn(2, 8, 128), mindspore.float32)
        input_tensor = Tensor(np.random.randn(2, 8, 128), mindspore.float32)
        output = model(hidden_states, input_tensor)
        assert output.shape == (2, 8, 128)

    def test_megatron_bert_layer(self):
        """
        test megatron_bert layer
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertLayer(config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (2, 32, 128)

    def test_megatron_bert_encoder(self):
        """
        test megatron_bert encoding
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertEncoder(config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (2, 32, 128)

    def test_megatron_bert_pooler(self):
        """
        test megatron_bert encoding
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertPooler(config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output[0].shape == (128,)

    def test_megatron_bert_prediction_head_transform(self):
        """
        test MegatronBertPredictionHeadTransform
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertPredictionHeadTransform(config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 32, 128)

    def test_megatron_bert_lm_prediction_head(self):
        """
        test MegatronBertLMPredictionHead
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertLMPredictionHead(config)

        hidden_states = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(hidden_states)
        assert output.shape == (2, 32, 100)

    def test_megatron_bert_only_mlm_head(self):
        """
        test MegatronBertOnlyMLMHead
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertOnlyMLMHead(config)

        sequence_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(sequence_output)
        assert output.shape == (2, 32, 100)

    def test_megatron_bert_only_nsp_head(self):
        """
        test MegatronBertOnlyNSPHead
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertOnlyNSPHead(config)

        pooled_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(pooled_output)
        assert output.shape == (2, 32, 2)

    def test_megatron_bert_pretraining_heads(self):
        """
        test MegatronBertPreTrainingHeads
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertPreTrainingHeads(config)

        sequence_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        pooled_output = Tensor(np.random.randn(2, 32, 128), mindspore.float32)
        output = model(sequence_output, pooled_output)
        assert output[0].shape == (2, 32, 100)
        assert output[1].shape == (2, 32, 2)

    def test_megatron_bert_model(self):
        """
        test MegatronBertModel
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertModel(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 32, 128)
        assert output[1].shape == (4, 128)

    def test_megatron_bert_for_pretraining(self):
        """
        test MegatronBertPreTraining
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)

        model = mindnlp.models.megatron_bert.MegatronBertForPreTraining(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 100)
        assert output[1].shape == (4, 2)

    def test_megatron_bert_for_causal_lm(self):
        """
        test MegatronBertForCausalLM
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        config.is_decoder = True
        model = mindnlp.models.megatron_bert.MegatronBertForCausalLM(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 100)
        assert len(output[1]) == 24
        assert output[1][0][0].shape == (4, 16, 16, 8)

    def test_megatron_bert_for_masked_lm(self):
        """
        test MegatronBertForMaskedLM
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertForMaskedLM(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 100)

    def test_megatron_bert_for_next_sentence_prediction(self):
        """
        test MegatronBertForNextSentencePrediction
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        config.is_decoder = True
        model = mindnlp.models.megatron_bert.MegatronBertForNextSentencePrediction(config)

        input_ids = Tensor(np.random.randint(0, 4, [2, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (2, 2)

    def test_megatron_bert_for_sequence_classification(self):
        """
        test MegatronBertForSequenceClassification
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        config.is_decoder = True
        model = mindnlp.models.megatron_bert.MegatronBertForSequenceClassification(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 2)

    def test_megatron_bert_for_multiple_choice(self):
        """
        test MegatronBertForMultipleChoice
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertForMultipleChoice(config)

        input_ids = Tensor(np.random.randint(0, 4, [1, 2, 4, 8]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 2)

    def test_megatron_bert_for_token_classification(self):
        """
        test MegatronBertForTokenClassification
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertForTokenClassification(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 16]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 16, 2)

    def test_megatron_bert_for_question_answering(self):
        """
        test MegatronBertForQuestionAnswering
        """
        config = mindnlp.models.megatron_bert.MegatronBertConfig(hidden_size=128, intermediate_size=128, vocab_size=100)
        model = mindnlp.models.megatron_bert.MegatronBertForQuestionAnswering(config)

        input_ids = Tensor(np.random.randint(0, 4, [4, 32]), mindspore.int32)
        output = model(input_ids)
        assert output[0].shape == (4, 32)
        assert output[1].shape == (4, 32)
