# coding=utf-8
# Copyright 2022 The Hugging Face Team.
#
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


import unittest

from mindnlp.transformers import MarkupLMConfig
from mindnlp.utils.testing_utils import require_mindspore, slow, is_mindspore_available
from mindnlp.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_mindspore_available():
    import mindspore
    from mindspore import nn, ops

    from mindnlp.transformers import (
        MarkupLMForQuestionAnswering,
        MarkupLMForSequenceClassification,
        MarkupLMForTokenClassification,
        MarkupLMModel,
    )

from mindnlp.transformers import MarkupLMFeatureExtractor, MarkupLMProcessor, MarkupLMTokenizer


class MarkupLMModelTester:
    """You can also import this e.g from .test_modeling_markuplm import MarkupLMModelTester"""

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
        max_xpath_tag_unit_embeddings=20,
        max_xpath_subs_unit_embeddings=30,
        tag_pad_id=2,
        subs_pad_id=2,
        max_depth=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        self.max_xpath_tag_unit_embeddings = max_xpath_tag_unit_embeddings
        self.max_xpath_subs_unit_embeddings = max_xpath_subs_unit_embeddings
        self.tag_pad_id = tag_pad_id
        self.subs_pad_id = subs_pad_id
        self.max_depth = max_depth

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        xpath_tags_seq = ids_tensor(
            [self.batch_size, self.seq_length, self.max_depth], self.max_xpath_tag_unit_embeddings
        )

        xpath_subs_seq = ids_tensor(
            [self.batch_size, self.seq_length, self.max_depth], self.max_xpath_subs_unit_embeddings
        )

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return (
            config,
            input_ids,
            xpath_tags_seq,
            xpath_subs_seq,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        )

    def get_config(self):
        return MarkupLMConfig(
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
            initializer_range=self.initializer_range,
            max_xpath_tag_unit_embeddings=self.max_xpath_tag_unit_embeddings,
            max_xpath_subs_unit_embeddings=self.max_xpath_subs_unit_embeddings,
            tag_pad_id=self.tag_pad_id,
            subs_pad_id=self.subs_pad_id,
            max_depth=self.max_depth,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        xpath_tags_seq,
        xpath_subs_seq,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        model = MarkupLMModel(config=config)
        model.set_train(False)
        print("Configs:", model.config.tag_pad_id, model.config.subs_pad_id)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        xpath_tags_seq,
        xpath_subs_seq,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        config.num_labels = self.num_labels
        model = MarkupLMForSequenceClassification(config)
        model.set_train(False)
        result = model(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        xpath_tags_seq,
        xpath_subs_seq,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        config.num_labels = self.num_labels
        model = MarkupLMForTokenClassification(config=config)
        model.set_train(False)
        result = model(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        xpath_tags_seq,
        xpath_subs_seq,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
    ):
        model = MarkupLMForQuestionAnswering(config=config)
        model.set_train(False)
        result = model(
            input_ids,
            xpath_tags_seq=xpath_tags_seq,
            xpath_subs_seq=xpath_subs_seq,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            xpath_tags_seq,
            xpath_subs_seq,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "xpath_tags_seq": xpath_tags_seq,
            "xpath_subs_seq": xpath_subs_seq,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }
        return config, inputs_dict


@require_mindspore
class MarkupLMModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            #MarkupLMModel,
            MarkupLMForSequenceClassification,
            MarkupLMForTokenClassification,
            MarkupLMForQuestionAnswering,
        )
        if is_mindspore_available()
        else None
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": MarkupLMModel,
            "question-answering": MarkupLMForQuestionAnswering,
            "text-classification": MarkupLMForSequenceClassification,
            "token-classification": MarkupLMForTokenClassification,
            "zero-shot": MarkupLMForSequenceClassification,
        }
        if is_mindspore_available()
        else {}
    )

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        # ValueError: Nodes must be of type `List[str]` (single pretokenized example), or `List[List[str]]`
        # (batch of pretokenized examples).
        return True

    def setUp(self):
        self.model_tester = MarkupLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MarkupLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)


def prepare_html_string():
    html_string = """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>

    <h1>This is a Heading</h1>
    <p>This is a paragraph.</p>

    </body>
    </html>
    """

    return html_string


@require_mindspore
class MarkupLMModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        # TODO use from_pretrained here
        feature_extractor = MarkupLMFeatureExtractor()
        tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base", from_pt = True)

        return MarkupLMProcessor(feature_extractor, tokenizer)

    @slow
    def test_forward_pass_no_head(self):
        model = MarkupLMModel.from_pretrained("microsoft/markuplm-base", from_pt = True)

        processor = self.default_processor

        inputs = processor(prepare_html_string(), return_tensors="ms")

        # forward pass
        with mindspore._no_grad():
            outputs = model(**inputs)

        # verify the last hidden states
        expected_shape = (1, 14, 768)
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)

        import numpy as np
        expected_slice = np.array(
            [[0.0675, -0.0052, 0.5001], [-0.2281, 0.0802, 0.2192], [-0.0583, -0.3311, 0.1185]]
        )
        self.assertTrue(np.allclose(outputs.last_hidden_state[0, :3, :3].numpy(), expected_slice, atol=1e-2))
