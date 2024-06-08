# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch WavLM model."""

import math
import unittest

import pytest
import numpy as np
from datasets import load_dataset

from mindnlp.transformers import WavLMConfig
from mindnlp.utils.testing_utils import require_mindspore, slow, is_mindspore_available
from mindnlp.modules.functional import normalize

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_mindspore_available():
    import mindspore
    from mindspore import ops

    from mindnlp.transformers import (
        Wav2Vec2FeatureExtractor,
        WavLMForAudioFrameClassification,
        WavLMForCTC,
        WavLMForSequenceClassification,
        WavLMForXVector,
        WavLMModel,
    )


class WavLMModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=16,
        feat_extract_norm="group",
        feat_extract_dropout=0.0,
        feat_extract_activation="gelu",
        conv_dim=(32, 32, 32),
        conv_stride=(4, 4, 4),
        conv_kernel=(8, 8, 8),
        conv_bias=False,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.1,  # this is most likely not correctly set yet
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        vocab_size=32,
        do_stable_layer_norm=False,
        tdnn_dim=(32, 32),
        tdnn_kernel=(3, 3),
        tdnn_dilation=(1, 1),
        xvector_output_dim=32,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.feat_extract_dropout = feat_extract_dropout
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.tdnn_dim = tdnn_dim
        self.tdnn_kernel = tdnn_kernel
        self.tdnn_dilation = tdnn_dilation
        self.xvector_output_dim = xvector_output_dim
        self.scope = scope

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return WavLMConfig(
            hidden_size=self.hidden_size,
            feat_extract_norm=self.feat_extract_norm,
            feat_extract_dropout=self.feat_extract_dropout,
            feat_extract_activation=self.feat_extract_activation,
            conv_dim=self.conv_dim,
            conv_stride=self.conv_stride,
            conv_kernel=self.conv_kernel,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            num_conv_pos_embedding_groups=self.num_conv_pos_embedding_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            tdnn_dim=self.tdnn_dim,
            tdnn_kernel=self.tdnn_kernel,
            tdnn_dilation=self.tdnn_dilation,
            xvector_output_dim=self.xvector_output_dim,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = WavLMModel(config=config)
        model.set_train(False)
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.output_seq_length, self.hidden_size)
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        # check: https://github.com/pytorch/fairseq/issues/3227
        model = WavLMModel(config=config)
        model.set_train(False)

        input_values = input_values[:3]
        attention_mask = ops.ones(input_values.shape, dtype=mindspore.bool_)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(input_values, attention_mask=attention_mask).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(np.allclose(output.asnumpy(), batch_output.asnumpy(), atol=1e-3))

    def check_ctc_loss(self, config, input_values, *args):
        model = WavLMForCTC(config=config)

        # make sure that dropout is disabled
        model.set_train(False)

        input_values = input_values[:3]
        attention_mask = ops.ones(input_values.shape, dtype=mindspore.int64)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(mindspore.Tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], min(max_length_labels).item() - 1), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_seq_classifier_loss(self, config, input_values, *args):
        model = WavLMForSequenceClassification(config=config)

        # make sure that dropout is disabled
        model.set_train(False)

        input_values = input_values[:3]
        attention_mask = ops.ones(input_values.shape, dtype=mindspore.int64)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        masked_loss = model(input_values, attention_mask=attention_mask, labels=labels).loss.item()
        unmasked_loss = model(input_values, labels=labels).loss.item()

        self.parent.assertTrue(isinstance(masked_loss, float))
        self.parent.assertTrue(isinstance(unmasked_loss, float))
        self.parent.assertTrue(masked_loss != unmasked_loss)

    def check_ctc_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = WavLMForCTC(config=config)
        model.set_train(True)

        # freeze feature encoder
        model.freeze_feature_encoder()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(mindspore.Tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels).item() - 2), model.config.vocab_size)

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lengths are at least
                # one shorter than logit lengths to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(ops.isinf(loss).item())

    def check_seq_classifier_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = WavLMForSequenceClassification(config=config)
        model.set_train(True)

        # freeze everything but the classification head
        model.freeze_base_model()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        labels = ids_tensor((input_values.shape[0], 1), len(model.config.id2label))

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(ops.isinf(loss).item())

    def check_output_attentions(self, config, input_values, attention_mask):
        model = WavLMModel(config=config)
        model.config.layerdrop = 1.0
        model.set_train(True)

        outputs = model(input_values, attention_mask=attention_mask, output_attentions=True)
        self.parent.assertTrue(len(outputs.attentions) > 0)

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = WavLMForCTC(config)
        model.set_train(True)

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(mindspore.Tensor(input_lengths))
        labels = ids_tensor((input_values.shape[0], max(max_length_labels).item() - 2), model.config.vocab_size + 100)

        with pytest.raises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_mindspore
class WavLMModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (WavLMForCTC, WavLMModel, WavLMForAudioFrameClassification, WavLMForSequenceClassification, WavLMForXVector)
        if is_mindspore_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "audio-classification": WavLMForSequenceClassification,
            "automatic-speech-recognition": WavLMForCTC,
            "feature-extraction": WavLMModel,
        }
        if is_mindspore_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = WavLMModelTester(self)
        self.config_tester = ConfigTester(self, config_class=WavLMConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_seq_classifier_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_seq_classifier_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_output_attentions(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_output_attentions(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # WavLM has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_values`
    def test_forward_signature(self):
        pass

    # WavLM cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # WavLM has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_common_attributes(self):
        pass

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.parameters_and_names():
                uniform_init_parms = [
                    "conv.weight",
                    "conv.parametrizations.weight",
                    "masked_spec_embed",
                    "codevectors",
                    "quantizer.weight_proj.weight",
                    "project_hid.weight",
                    "project_hid.bias",
                    "project_q.weight",
                    "project_q.bias",
                    "feature_projection.projection.weight",
                    "feature_projection.projection.bias",
                    "label_embeddings_concat",
                    "rel_attn_embed",
                    "objective.weight",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill(3)
        if hasattr(module, "codevectors") and module.codevectors is not None:
            module.codevectors.data.fill(3)
        if hasattr(module, "masked_spec_embed") and module.masked_spec_embed is not None:
            module.masked_spec_embed.data.fill(3)

    @unittest.skip(reason="Feed forward chunking is not implemented for WavLM")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.assertIsNotNone(model)


@require_mindspore
@slow
class WavLMModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def _load_superb(self, task, num_samples):
        ds = load_dataset("anton-l/superb_dummy", task, split="test")

        return ds[:num_samples]

    def test_inference_base(self):
        model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus", return_attention_mask=True
        )

        input_speech = self._load_datasamples(2)

        inputs = feature_extractor(input_speech, return_tensors="ms", padding=True)

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask

        hidden_states_slice = (
            model(input_values, attention_mask=attention_mask).last_hidden_state[:, -2:, -2:]
        )

        EXPECTED_HIDDEN_STATES_SLICE = mindspore.Tensor(
            [[[0.0577, 0.1161], [0.0579, 0.1165]], [[0.0199, 0.1237], [0.0059, 0.0605]]]
        )
        self.assertTrue(np.allclose(hidden_states_slice.asnumpy(), EXPECTED_HIDDEN_STATES_SLICE.asnumpy(), atol=5e-2))

    def test_inference_large(self):
        model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large", return_attention_mask=True
        )

        input_speech = self._load_datasamples(2)

        inputs = feature_extractor(input_speech, return_tensors="ms", padding=True)

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask

        hidden_states_slice = (
            model(input_values, attention_mask=attention_mask).last_hidden_state[:, -2:, -2:]
        )

        EXPECTED_HIDDEN_STATES_SLICE = mindspore.Tensor(
            [[[0.2122, 0.0500], [0.2118, 0.0563]], [[0.1353, 0.1818], [0.2453, 0.0595]]]
        )

        self.assertTrue(np.allclose(hidden_states_slice.asnumpy(), EXPECTED_HIDDEN_STATES_SLICE.asnumpy(), rtol=5e-2))

    def test_inference_diarization(self):
        model = WavLMForAudioFrameClassification.from_pretrained("microsoft/wavlm-base-plus-sd")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sd")
        input_data = self._load_superb("sd", 4)
        inputs = processor(input_data["speech"], return_tensors="ms", padding=True, sampling_rate=16_000)

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask
        outputs = model(input_values, attention_mask=attention_mask)
        # labels is a one-hot array of shape (num_frames, num_speakers)
        labels = (outputs.logits > 0).astype(mindspore.int64)

        # s3prl logits for the same batch
        expected_logits = mindspore.Tensor(
            [
                [[-5.9566, -8.6554], [-5.7137, -8.9386], [-5.7906, -7.0973], [-5.7829, -5.9999]],
                [[-5.2086, -7.7878], [-4.8890, -7.9312], [-4.2004, -3.9101], [-5.4480, -4.6932]],
                [[-4.6105, -6.7178], [-5.1930, -6.1635], [-2.6228, -4.1123], [-2.7646, -3.1576]],
                [[-4.4477, -7.9206], [-3.9339, -7.3707], [-4.9528, -4.8242], [-3.6921, -2.9687]],
            ],
        )
        self.assertEqual(labels[0, :, 0].sum(), 258)
        self.assertEqual(labels[0, :, 1].sum(), 647)
        self.assertTrue(np.allclose(outputs.logits[:, :4].asnumpy(), expected_logits.asnumpy(), atol=1e-2))

    def test_inference_speaker_verification(self):
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        input_data = self._load_superb("si", 4)

        inputs = processor(input_data["speech"], return_tensors="ms", padding=True)
        labels = mindspore.Tensor([5, 1, 1, 3]).T

        input_values = inputs.input_values
        attention_mask = inputs.attention_mask
        outputs = model(input_values, attention_mask=attention_mask, labels=labels)
        embeddings = outputs.embeddings / ops.norm(outputs.embeddings, dim=-1, keepdim=True)

        cosine_sim = lambda x, y: ops.cosine_similarity(x, y, dim=-1)
        # id10002 vs id10002
        self.assertAlmostEqual(cosine_sim(embeddings[1], embeddings[2]).item(), 0.9787, 3)
        # id10006 vs id10002
        self.assertAlmostEqual(cosine_sim(embeddings[0], embeddings[1]).item(), 0.5064, 3)
        # id10002 vs id10004
        self.assertAlmostEqual(cosine_sim(embeddings[2], embeddings[3]).item(), 0.4780, 3)

        self.assertAlmostEqual(outputs.loss.item(), 18.4154, 2)