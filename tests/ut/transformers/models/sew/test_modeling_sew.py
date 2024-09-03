# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Testing suite for the MindSpore Hubert model."""

import math
import unittest

import pytest

import numpy as np

from mindnlp.utils.testing_utils import (
    require_soundfile,
    is_mindspore_available,
    slow,
    require_mindspore,
)

from mindnlp.transformers import SEWConfig

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
        SEWForCTC,
        SEWForSequenceClassification,
        SEWModel,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )
    from mindnlp.transformers.models.hubert.modeling_hubert import _compute_mask_indices


class SEWModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=1024,  # speech is longer
        is_training=False,
        hidden_size=32,
        feat_extract_norm="group",
        feat_extract_dropout=0.0,
        feat_extract_activation="gelu",
        conv_dim=(64, 32, 32),
        conv_stride=(5, 2, 1),
        conv_kernel=(10, 3, 1),
        conv_bias=False,
        num_conv_pos_embeddings=31,
        num_conv_pos_embedding_groups=2,
        squeeze_factor=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout=0.1,
        intermediate_size=20,
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        initializer_range=0.02,
        vocab_size=32,
        do_stable_layer_norm=False,
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
        self.squeeze_factor = squeeze_factor
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout = hidden_dropout
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.scope = scope

        output_seq_length = self.seq_length
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            output_seq_length = (output_seq_length - (kernel - 1)) / stride
        self.output_seq_length = int(math.ceil(output_seq_length))
        self.encoder_seq_length = self.output_seq_length // self.squeeze_factor

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_values, attention_mask

    def get_config(self):
        return SEWConfig(
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
            squeeze_factor=self.squeeze_factor,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_dropout=self.hidden_dropout,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            hidden_act=self.hidden_act,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model(self, config, input_values, attention_mask):
        model = SEWModel(config=config)
        model.set_train(False)
        result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.output_seq_length, self.hidden_size),
        )

    def create_and_check_batch_inference(self, config, input_values, *args):
        # test does not pass for models making use of `group_norm`
        model = SEWModel(config=config)
        model.set_train(False)

        input_values = input_values[:3]
        attention_mask = ops.ones(input_values.shape, dtype=mindspore.bool_)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0.0

        batch_outputs = model(
            input_values, attention_mask=attention_mask
        ).last_hidden_state

        for i in range(input_values.shape[0]):
            input_slice = input_values[i : i + 1, : input_lengths[i]]
            output = model(input_slice).last_hidden_state

            batch_output = batch_outputs[i : i + 1, : output.shape[1]]
            self.parent.assertTrue(
                np.allclose(output.asnumpy(), batch_output.asnumpy(), atol=1e-3)
            )

    def check_ctc_loss(self, config, input_values, *args):
        model = SEWForCTC(config=config)

        # make sure that dropout is disabled
        model.set_train(False)

        input_values = input_values[:3]
        attention_mask = ops.ones(input_values.shape, dtype=mindspore.int64)

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(
            mindspore.tensor(input_lengths)
        )
        labels = ids_tensor(
            (input_values.shape[0], int((min(max_length_labels) - 1).asnumpy())),
            model.config.vocab_size,
        )

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0
            attention_mask[i, input_lengths[i] :] = 0

        model.config.ctc_loss_reduction = "sum"
        sum_loss = model(
            input_values, attention_mask=attention_mask, labels=labels
        ).loss.item()

        model.config.ctc_loss_reduction = "mean"
        mean_loss = model(
            input_values, attention_mask=attention_mask, labels=labels
        ).loss.item()

        self.parent.assertTrue(isinstance(sum_loss, float))
        self.parent.assertTrue(isinstance(mean_loss, float))

    def check_ctc_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = SEWForCTC(config=config)
        model.set_train()

        # freeze feature encoder
        model.freeze_feature_encoder()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(
            mindspore.tensor(input_lengths)
        )
        labels = ids_tensor(
            (input_values.shape[0], int((max(max_length_labels) - 2).asnumpy())),
            model.config.vocab_size,
        )

        # pad input
        for i in range(len(input_lengths)):
            input_values[i, input_lengths[i] :] = 0.0

            if max_length_labels[i] < labels.shape[-1]:
                # it's important that we make sure that target lengths are at least
                # one shorter than logit lengths to prevent -inf
                labels[i, max_length_labels[i] - 1 :] = -100

        loss = model(input_values, labels=labels).loss
        self.parent.assertFalse(ops.isinf(loss).item())

    def check_seq_classifier_loss(self, config, input_values, *args):
        model = SEWForSequenceClassification(config=config)

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

        masked_loss = model(
            input_values, attention_mask=attention_mask, labels=labels
        ).loss.item()
        unmasked_loss = model(input_values, labels=labels).loss.item()
        self.parent.assertTrue(isinstance(masked_loss, float))
        self.parent.assertTrue(isinstance(unmasked_loss, float))
        self.parent.assertTrue(masked_loss != unmasked_loss)

    def check_seq_classifier_training(self, config, input_values, *args):
        config.ctc_zero_infinity = True
        model = SEWForSequenceClassification(config=config)
        model.set_train()

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

    def check_labels_out_of_vocab(self, config, input_values, *args):
        model = SEWForCTC(config)
        model.set_train()

        input_values = input_values[:3]

        input_lengths = [input_values.shape[-1] // i for i in [4, 2, 1]]
        max_length_labels = model._get_feat_extract_output_lengths(
            mindspore.tensor(input_lengths)
        )
        labels = ids_tensor(
            (input_values.shape[0], int((max(max_length_labels) - 2).asnumpy())),
            model.config.vocab_size + 100,
        )

        with pytest.raises(ValueError):
            model(input_values, labels=labels)

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {"input_values": input_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_mindspore
class SEWModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (SEWForCTC, SEWModel, SEWForSequenceClassification)
        if is_mindspore_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "audio-classification": SEWForSequenceClassification,
            "automatic-speech-recognition": SEWForCTC,
            "feature-extraction": SEWModel,
        }
        if is_mindspore_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False

    def setUp(self):
        self.model_tester = SEWModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SEWConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_ctc_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_loss(*config_and_inputs)

    def test_ctc_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_ctc_training(*config_and_inputs)

    def test_labels_out_of_vocab(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_labels_out_of_vocab(*config_and_inputs)

    # Hubert has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # `input_ids` is renamed to `input_values`
    def test_forward_signature(self):
        pass

    # SEW cannot resize token embeddings
    # since it has no tokens embeddings
    def test_resize_tokens_embeddings(self):
        pass

    # SEW has no inputs_embeds
    # and thus the `get_input_embeddings` fn
    # is not implemented
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip("No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip("No support for low_cpu_mem_usage=True.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass

    @unittest.skip("mindnlp does not support `_attn_implementation` config.")
    def test_seq_classifier_loss_inference(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_loss(*config_and_inputs)

    def test_seq_classifier_train(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.check_seq_classifier_training(*config_and_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.parameters_and_names():
                uniform_init_parms = [
                    "conv.weight",
                    "masked_spec_embed",
                    "quantizer.weight_proj.weight",
                ]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0
                            <= ((param.data.mean() * 1e9).round() / 1e9).item()
                            <= 1.0,
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
            module.weight.data.fill_(3)
        if hasattr(module, "weight_g") and module.weight_g is not None:
            module.weight_g.data.fill_(3)
        if hasattr(module, "weight_v") and module.weight_v is not None:
            module.weight_v.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if (
            hasattr(module, "masked_spec_embed")
            and module.masked_spec_embed is not None
        ):
            module.masked_spec_embed.data.fill_(3)

    @unittest.skip(reason="Feed forward chunking is not implemented")
    def test_feed_forward_chunking(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = SEWModel.from_pretrained("asapp/sew-tiny-100k")
        self.assertIsNotNone(model)


@require_mindspore
class SEWUtilsTest(unittest.TestCase):
    def test_compute_mask_indices(self):
        batch_size = 4
        sequence_length = 60
        mask_prob = 0.5
        mask_length = 1

        mask = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob, mask_length
        )
        mask = mindspore.Tensor(mask)

        self.assertListEqual(
            mask.sum(axis=-1).tolist(),
            [mask_prob * sequence_length for _ in range(batch_size)],
        )

    def test_compute_mask_indices_overlap(self):
        batch_size = 4
        sequence_length = 80
        mask_prob = 0.5
        mask_length = 4

        mask = _compute_mask_indices(
            (batch_size, sequence_length), mask_prob, mask_length
        )
        mask = mindspore.Tensor(mask)

        # because of overlap mask don't have to add up exactly to `mask_prob * sequence_length`, but have to be smaller or equal
        for batch_sum in mask.sum(axis=-1):
            self.assertTrue(int(batch_sum) <= mask_prob * sequence_length)


@require_mindspore
@require_soundfile
@slow
class SEWModelIntegrationTest(unittest.TestCase):
    def _load_datasamples(self, num_samples):
        from datasets import load_dataset

        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        # automatic decoding with librispeech
        speech_samples = ds.sort("id").filter(
            lambda x: x["id"] in [f"1272-141231-000{i}" for i in range(num_samples)]
        )[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    def test_inference_pretrained_batched(self):
        model = SEWModel.from_pretrained("asapp/sew-tiny-100k")
        processor = Wav2Vec2FeatureExtractor.from_pretrained("asapp/sew-tiny-100k")

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="ms", padding=True)

        input_values = inputs.input_values

        outputs = model(input_values).last_hidden_state

        # expected outputs taken from the original SEW implementation
        expected_outputs_first = mindspore.tensor(
            [
                [
                    [0.1509, 0.5372, 0.3061, -0.1694],
                    [-0.1700, 0.5764, 0.2753, -0.1299],
                    [0.1281, 0.7949, 0.2342, -0.1624],
                    [-0.1627, 0.6710, 0.2215, -0.1317],
                ],
                [
                    [0.0408, 1.4355, 0.8605, -0.0968],
                    [0.0393, 1.2368, 0.6826, 0.0364],
                    [-0.1269, 1.9215, 1.1677, -0.1297],
                    [-0.1654, 1.6524, 0.6877, -0.0196],
                ],
            ],
        )
        expected_outputs_last = mindspore.tensor(
            [
                [
                    [1.3379, -0.1450, -0.1500, -0.0515],
                    [0.8364, -0.1680, -0.1248, -0.0689],
                    [1.2791, -0.1507, -0.1523, -0.0564],
                    [0.8208, -0.1690, -0.1199, -0.0751],
                ],
                [
                    [0.6959, -0.0861, -0.1235, -0.0861],
                    [0.4700, -0.1686, -0.1141, -0.1199],
                    [1.0776, -0.1137, -0.0124, -0.0472],
                    [0.5774, -0.1675, -0.0376, -0.0823],
                ],
            ],
        )
        expected_output_sum = 62146.7422

        self.assertTrue(
            np.allclose(
                outputs[:, :4, :4].asnumpy(),
                expected_outputs_first.asnumpy(),
                atol=5e-2,
            )
        )
        self.assertTrue(
            np.allclose(
                outputs[:, -4:, -4:].asnumpy(),
                expected_outputs_last.asnumpy(),
                atol=5e-2,
            )
        )
        self.assertTrue(abs(outputs.sum() - expected_output_sum) < 5)

    def test_inference_ctc_batched(self):
        model = SEWForCTC.from_pretrained("asapp/sew-tiny-100k-ft-ls100h")
        processor = Wav2Vec2Processor.from_pretrained(
            "asapp/sew-tiny-100k-ft-ls100h", do_lower_case=True
        )

        input_speech = self._load_datasamples(2)

        inputs = processor(input_speech, return_tensors="ms", padding=True)

        input_values = inputs.input_values

        logits = model(input_values).logits

        predicted_ids = ops.argmax(logits, dim=-1)
        predicted_trans = processor.batch_decode(predicted_ids)

        EXPECTED_TRANSCRIPTIONS = [
            "a man said to the universe sir i exist",
            "swet covered brian's body trickling into the tightloine closs hat was the only garment he wore",
        ]
        self.assertListEqual(predicted_trans, EXPECTED_TRANSCRIPTIONS)
