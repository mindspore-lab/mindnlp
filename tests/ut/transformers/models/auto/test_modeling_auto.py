# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import copy
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import pytest

from mindnlp.transformers import BertConfig, GPT2Model
from mindnlp.transformers.models.auto.configuration_auto import CONFIG_MAPPING
from mindnlp.utils import is_mindspore_available, is_safetensors_available
from mindnlp.utils.testing_utils import (
    DUMMY_UNKNOWN_IDENTIFIER,
    SMALL_MODEL_IDENTIFIER,
    RequestCounter,
    require_mindspore,
    slow
)

from ..bert.test_modeling_bert import BertModelTester


sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from test_module.custom_configuration import CustomConfig  # noqa E402


if is_mindspore_available():
    import mindspore
    from mindspore import ops
    from test_module.custom_modeling import CustomModel

    from mindnlp.transformers import (
        AutoBackbone,
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForPreTraining,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoModelForTableQuestionAnswering,
        AutoModelForTokenClassification,
        AutoModelWithLMHead,
        BertForMaskedLM,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        BertModel,
        GPT2Config,
        GPT2LMHeadModel,
        RobertaForMaskedLM,
        T5Config,
        T5ForConditionalGeneration,
    )
    from mindnlp.transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING,
        MODEL_FOR_MASKED_LM_MAPPING,
        MODEL_FOR_PRETRAINING_MAPPING,
        MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        MODEL_MAPPING,
    )
    from mindnlp.transformers.models.bert.modeling_bert import BERT_SUPPORT_LIST
    from mindnlp.transformers.models.gpt2.modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    from mindnlp.transformers.models.t5.modeling_t5 import T5_PRETRAINED_MODEL_ARCHIVE_LIST


@require_mindspore
class AutoModelTest(unittest.TestCase):

    @slow
    def test_model_from_pretrained(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModel.from_pretrained(model_name)
            model, loading_info = AutoModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertModel)

            self.assertEqual(len(loading_info["missing_keys"]), 0)
            # When using PyTorch checkpoint, the expected value is `8`. With `safetensors` checkpoint (if it is
            # installed), the expected value becomes `7`.
            EXPECTED_NUM_OF_UNEXPECTED_KEYS = 7 if is_safetensors_available() else 8
            self.assertEqual(len(loading_info["unexpected_keys"]), EXPECTED_NUM_OF_UNEXPECTED_KEYS)
            # self.assertEqual(len(loading_info["mismatched_keys"]), 0)
            # self.assertEqual(len(loading_info["error_msgs"]), 0)


    @slow
    def test_model_for_pretraining_from_pretrained(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForPreTraining.from_pretrained(model_name)
            model, loading_info = AutoModelForPreTraining.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForPreTraining)
            # Only one value should not be initialized and in the missing keys.
            for key, value in loading_info.items():
                self.assertEqual(len(value), 0)

    @slow
    def test_lmhead_model_from_pretrained(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelWithLMHead.from_pretrained(model_name)
            model, loading_info = AutoModelWithLMHead.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)


    @slow
    def test_model_for_causal_lm(self):
        for model_name in GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, GPT2Config)

            model = AutoModelForCausalLM.from_pretrained(model_name)
            model, loading_info = AutoModelForCausalLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, GPT2LMHeadModel)


    @slow
    def test_model_for_masked_lm(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model, loading_info = AutoModelForMaskedLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForMaskedLM)

    @slow
    def test_model_for_encoder_decoder_lm(self):
        for model_name in T5_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, T5Config)

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, T5ForConditionalGeneration)


    @slow
    def test_sequence_classification_model_from_pretrained(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model, loading_info = AutoModelForSequenceClassification.from_pretrained(
                model_name, output_loading_info=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForSequenceClassification)


    @slow
    def test_question_answering_model_from_pretrained(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            model, loading_info = AutoModelForQuestionAnswering.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForQuestionAnswering)


    @unittest.skip('not support yet')
    @slow
    def test_table_question_answering_model_from_pretrained(self):
        for model_name in TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST[5:6]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, TapasConfig)

            model = AutoModelForTableQuestionAnswering.from_pretrained(model_name)
            model, loading_info = AutoModelForTableQuestionAnswering.from_pretrained(
                model_name, output_loading_info=True
            )
            self.assertIsNotNone(model)
            self.assertIsInstance(model, TapasForQuestionAnswering)


    @slow
    def test_token_classification_model_from_pretrained(self):
        for model_name in BERT_SUPPORT_LIST[:1]:
            config = AutoConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, BertConfig)

            model = AutoModelForTokenClassification.from_pretrained(model_name)
            model, loading_info = AutoModelForTokenClassification.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, BertForTokenClassification)


    def test_from_pretrained_identifier(self):
        model = AutoModelWithLMHead.from_pretrained(SMALL_MODEL_IDENTIFIER)
        self.assertIsInstance(model, BertForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    def test_from_identifier_from_model_type(self):
        model = AutoModelWithLMHead.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER)
        self.assertIsInstance(model, RobertaForMaskedLM)
        self.assertEqual(model.num_parameters(), 14410)
        self.assertEqual(model.num_parameters(only_trainable=True), 14410)

    @unittest.skip('not support yet')
    def test_from_pretrained_with_tuple_values(self):
        # For the auto model mapping, FunnelConfig has two models: FunnelModel and FunnelBaseModel
        model = AutoModel.from_pretrained("sgugger/funnel-random-tiny")
        self.assertIsInstance(model, FunnelModel)

        config = copy.deepcopy(model.config)
        config.architectures = ["FunnelBaseModel"]
        model = AutoModel.from_config(config)
        self.assertIsInstance(model, FunnelBaseModel)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModel.from_pretrained(tmp_dir)
            self.assertIsInstance(model, FunnelBaseModel)

    def test_new_model_registration(self):
        AutoConfig.register("custom", CustomConfig)

        auto_classes = [
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        ]

        try:
            for auto_class in auto_classes:
                with self.subTest(auto_class.__name__):
                    # Wrong config class will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, CustomModel)
                    auto_class.register(CustomConfig, CustomModel)
                    # Trying to register something existing in the Transformers library will raise an error
                    with self.assertRaises(ValueError):
                        auto_class.register(BertConfig, BertModel)

                    # Now that the config is registered, it can be used as any other config with the auto-API
                    tiny_config = BertModelTester(self).get_config()
                    config = CustomConfig(**tiny_config.to_dict())
                    model = auto_class.from_config(config)
                    self.assertIsInstance(model, CustomModel)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        model.save_pretrained(tmp_dir)
                        new_model = auto_class.from_pretrained(tmp_dir)
                        # The model is a CustomModel but from the new dynamically imported class.
                        self.assertIsInstance(new_model, CustomModel)

        finally:
            if "custom" in CONFIG_MAPPING._extra_content:
                del CONFIG_MAPPING._extra_content["custom"]
            for mapping in (
                MODEL_MAPPING,
                MODEL_FOR_PRETRAINING_MAPPING,
                MODEL_FOR_QUESTION_ANSWERING_MAPPING,
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_MASKED_LM_MAPPING,
            ):
                if CustomConfig in mapping._extra_content:
                    del mapping._extra_content[CustomConfig]

    def test_repo_not_found(self):
        with self.assertRaises(
            EnvironmentError
        ):
            _ = AutoModel.from_pretrained("bert-base")

    def test_model_file_not_found(self):
        with self.assertRaises(
            EnvironmentError,
        ):
            _ = AutoModel.from_pretrained("hf-internal-testing/config-no-model")

    # def test_cached_model_has_minimum_calls_to_head(self):
    #     # Make sure we have cached the model.
    #     _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
    #     with RequestCounter() as counter:
    #         _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
    #     self.assertEqual(counter["GET"], 0)
    #     self.assertEqual(counter["HEAD"], 1)
    #     self.assertEqual(counter.total_calls, 1)

    #     # With a sharded checkpoint
    #     _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
    #     with RequestCounter() as counter:
    #         _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
    #     self.assertEqual(counter["GET"], 0)
    #     self.assertEqual(counter["HEAD"], 1)
    #     self.assertEqual(counter.total_calls, 1)

    def test_attr_not_existing(self):
        from mindnlp.transformers.models.auto.auto_factory import _LazyAutoMapping

        _CONFIG_MAPPING_NAMES = OrderedDict([("bert", "BertConfig")])
        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "GhostModel")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)

        with pytest.raises(ValueError, match=r"Could not find GhostModel neither in .* nor in .*!"):
            _MODEL_MAPPING[BertConfig]

        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "BertModel")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)
        self.assertEqual(_MODEL_MAPPING[BertConfig], BertModel)

        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "GPT2Model")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)
        self.assertEqual(_MODEL_MAPPING[BertConfig], GPT2Model)