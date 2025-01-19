# tests/test_modeling_mimi.py

import unittest
import mindspore
from mindspore import ops
from mindnlp.models.mimi import MimiModel, MimiConfig

class MimiModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = MimiModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MimiConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_test_model(*config_and_inputs)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_test_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_masked_lm(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_multiple_choice(*config_and_inputs)

    def test_for_next_sequence_prediction(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_next_sequence_prediction(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_test_for_token_classification(*config_and_inputs)

    @unittest.skip(reason="Not implemented yet")
    def test_model_as_encoder_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_encoder_decoder()
        self.model_tester.create_and_test_model(*config_and_inputs)

class MimiModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 13
        self.seq_length = 7
        self.is_training = True
        self.use_input_mask = True
        self.use_token_type_ids = True
        self.use_labels = True
        self.vocab_size = 99
        self.hidden_size = 32
        self.num_hidden_layers = 5
        self.num_attention_heads = 4
        self.intermediate_size = 37
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 16
        self.type_sequence_label_size = 2
        self.initializer_range = 0.02
        self.num_choices = 4

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = ops.ones((self.batch_size, self.seq_length), dtype=ops.float32)
        token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_choices)

        config = MimiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels

    def prepare_config_and_inputs_for_decoder(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = ops.ones((self.batch_size, self.seq_length), dtype=ops.float32)
        token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_choices)

        config = MimiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
        )

        return config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels

    def create_and_test_model(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiModel(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        outputs = model(input_ids, attention_mask=input_mask)
        outputs = model(input_ids)

        result = outputs.last_hidden_state
        self.parent.assertEqual(result.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_test_for_masked_lm(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForMaskedLM(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_test_for_multiple_choice(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForMultipleChoice(config=config)
        model.set_train(self.is_training)
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1)
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1)
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1)
        outputs = model(multiple_choice_inputs_ids, attention_mask=multiple_choice_input_mask, token_type_ids=multiple_choice_token_type_ids, labels=sequence_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, self.num_choices))

    def create_and_test_for_next_sequence_prediction(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForNextSentencePrediction(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, 2))

    def create_and_test_for_pretraining(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForPreTraining(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels, next_sentence_label=sequence_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_test_for_question_answering(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForQuestionAnswering(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, 2))

    def create_and_test_for_sequence_classification(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForSequenceClassification(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_test_for_token_classification(self, config, input_ids, input_mask, token_type_ids, sequence_labels, token_labels):
        model = MimiForTokenClassification(config=config)
        model.set_train(self.is_training)
        outputs = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        result = outputs.logits
        self.parent.assertEqual(result.shape, (self.batch_size, self.seq_length, self.num_choices))

def ids_tensor(shape, vocab_size):
    return ops.randint(0, vocab_size, shape)

if __name__ == "__main__":
    unittest.main()
