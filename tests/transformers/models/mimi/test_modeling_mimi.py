# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Mindnlp Mimi model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
from datasets import Audio, load_dataset
from parameterized import parameterized

from mindnlp.transformers import AutoFeatureExtractor
from mindnlp.transformers.models.mimi import MimiConfig
from mindnlp.utils.testing_utils import (
    is_flaky,
    is_mindspore_available,
    require_mindspore,
    slow,
)
from mindnlp.core.autograd import no_grad
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_mindspore_available():
    import mindspore
    from mindspore import ops

    from mindnlp.transformers.models.mimi import MimiModel


# Copied from transformers.tests.encodec.test_modeling_encodec.prepare_inputs_dict
def prepare_inputs_dict(
    config,
    input_ids=None,
    input_values=None,
    decoder_input_ids=None,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if input_ids is not None:
        encoder_dict = {"input_ids": input_ids}
    else:
        encoder_dict = {"input_values": input_values}

    decoder_dict = {
        "decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}

    return {**encoder_dict, **decoder_dict}


@require_mindspore
class MimiModelTester:
    def __init__(
        self,
        parent,
        batch_size=5,
        num_channels=1,
        is_training=False,
        intermediate_size=40,
        hidden_size=32,
        num_filters=8,
        num_residual_layers=1,
        upsampling_ratios=[8, 4],
        codebook_size=64,
        vector_quantization_hidden_dimension=64,
        codebook_dim=64,
        upsample_groups=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        sliding_window=4,
        use_cache=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios
        self.codebook_size = codebook_size
        self.vector_quantization_hidden_dimension = vector_quantization_hidden_dimension
        self.codebook_dim = codebook_dim
        self.upsample_groups = upsample_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = sliding_window
        self.use_cache = use_cache

    def prepare_config_and_inputs(self):
        input_values = floats_tensor(
            [self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        import mindspore
        config, inputs_dict = self.prepare_config_and_inputs()
        inputs_dict["audio_codes"] = ids_tensor([self.batch_size, 1, self.num_channels], self.codebook_size).type(
            mindspore.int32
        )

        return config, inputs_dict

    def get_config(self):
        return MimiConfig(
            audio_channels=self.num_channels,
            chunk_in_sec=None,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            num_residual_layers=self.num_residual_layers,
            upsampling_ratios=self.upsampling_ratios,
            codebook_size=self.codebook_size,
            vector_quantization_hidden_dimension=self.vector_quantization_hidden_dimension,
            upsample_groups=self.upsample_groups,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            sliding_window=self.sliding_window,
            codebook_dim=self.codebook_dim,
            use_cache=self.use_cache,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        model = MimiModel(config=config).eval()

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size,
                                        self.num_channels, self.intermediate_size)
        )


@require_mindspore
class MimiModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MimiModel,) if is_mindspore_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    test_torchscript = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does support returning hidden states
        inputs_dict = super()._prepare_for_class(
            inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = MimiModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MimiConfig, hidden_size=37, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @require_mindspore
    def test_model_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values",
                                  "padding_mask", "num_quantizers"]
            self.assertListEqual(
                arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip(reason="The MimiModel does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="The MimiModel does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `hidden_states` logic")
    def test_torchscript_output_hidden_state(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest._create_and_check_torchscript
    def _create_and_check_torchscript(self, config, inputs_dict):
        import mindspore
        if not self.test_torchscript:
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(
            config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class)

            main_input_name = model_class.main_input_name

            try:
                main_input = inputs[main_input_name]
                model(main_input)
                traced_model = mindspore.jit.trace(model, main_input)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    mindspore.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = mindspore.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.eval()

            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()),
                             set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if ops.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if ops.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                if layer_name in loaded_model_state_dict:
                    p2 = loaded_model_state_dict[layer_name]
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

            self.assertTrue(models_equal)

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_determinism
    @require_mindspore
    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # outputs are not tensors but list (since each sequence don't have the same frame_length)
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()
            with no_grad():
                first = model(
                    **self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(
                    **self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_model_outputs_equivalence
    @require_mindspore
    def test_model_outputs_equivalence(self):
        import mindspore
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            def allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08):
                """
                Checks if all elements of two tensors are close within a tolerance.
                """
                tensor1 = tensor1.astype(mindspore.float32)
                tensor2 = tensor2.astype(mindspore.float32)
                diff = ops.abs(tensor1 - tensor2)
                return ops.all(diff <= (atol + rtol * ops.abs(tensor2)))

            with no_grad():
                tuple_output = model(
                    **tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(
                    **dict_inputs, return_dict=True, **additional_kwargs)

                self.assertTrue(isinstance(tuple_output, tuple))
                self.assertTrue(isinstance(dict_output, dict))

                for tuple_value, dict_value in zip(tuple_output, dict_output.values()):
                    self.assertTrue(
                        allclose(
                            set_nan_tensor_to_zero(tuple_value), set_nan_tensor_to_zero(dict_value), atol=1e-5
                        )
                    )

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.eval()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                uniform_init_parms = ["conv", "input_proj", "output_proj"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() *
                                     1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_identity_shortcut
    @require_mindspore
    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)

    # Overwrite to use `audio_values` as the tensors to compare.
    # TODO: Try to do this in the parent class.
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @unittest.skip("no SDPA")
    @unittest.skip("no flash_attn")
    @slow
    @is_flaky()
    @unittest.skip(reason="The MimiModel does not support right padding")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="The MimiModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @is_flaky()
    @require_mindspore
    def test_batching_equivalence(self):
        super().test_batching_equivalence()


# Copied from transformers.tests.encodec.test_modeling_encodec.normalize
def normalize(arr):
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr

# Copied from transformers.tests.encodec.test_modeling_encodec.compute_rmse


def compute_rmse(arr1, arr2):
    arr1_normalized = normalize(arr1)
    arr2_normalized = normalize(arr2)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_mindspore
class MimiIntegrationTest(unittest.TestCase):
    def test_integration_using_cache_decode(self):
        import mindspore
        expected_rmse = {
            "8": 0.0018785292,
            "32": 0.0012330565,
        }

        librispeech_dummy = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        model = MimiModel.from_pretrained(model_id, use_cache=True).to(
            mindspore.get_context('device_target'))
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column(
            "audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(mindspore.get_context('device_target'))

        for num_codebooks, expected_rmse in expected_rmse.items():
            with no_grad():
                # use max bandwith for best possible reconstruction
                encoder_outputs = model.encode(
                    inputs["input_values"], num_quantizers=int(num_codebooks))

                audio_codes = encoder_outputs[0]

                decoder_outputs_first_part = model.decode(
                    audio_codes[:, :, : audio_codes.shape[2] // 2])
                decoder_outputs_second_part = model.decode(
                    audio_codes[:, :, audio_codes.shape[2] // 2:],
                    decoder_past_key_values=decoder_outputs_first_part.decoder_past_key_values,
                )

                audio_output_entire_context = model.decode(audio_codes)[0]
                audio_output_concat_context = mindspore.ops.cat(
                    [decoder_outputs_first_part[0],
                        decoder_outputs_second_part[0]]
                )

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(
                audio_output_concat_context.squeeze().numpy(),
                audio_output_entire_context.squeeze().numpy(),
            )
            self.assertTrue(rmse < 1e-3)

    def test_integration(self):
        import mindspore
        expected_rmses = {
            "8": 0.0018785292,
            "32": 0.0012330565,
        }
        expected_codesums = {
            "8": 430423,
            "32": 1803071,
        }
        librispeech_dummy = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column(
            "audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        ).to(mindspore.get_context('device_target'))

        def allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08):
            """
            Checks if all elements of two tensors are close within a tolerance.
            """
            diff = ops.abs(tensor1 - tensor2)
            return ops.all(diff <= (atol + rtol * ops.abs(tensor2)))

        for use_cache in [False, True]:
            model = MimiModel.from_pretrained(model_id, use_cache=use_cache).to(
                mindspore.get_context('device_target'))
            for num_codebooks, expected_rmse in expected_rmses.items():
                with no_grad():
                    # use max bandwith for best possible reconstruction
                    encoder_outputs = model.encode(
                        inputs["input_values"], num_quantizers=int(num_codebooks))

                    audio_code_sums = encoder_outputs[0].sum().item()

                    # make sure audio encoded codes are correct
                    # assert relative difference less than a threshold, because `audio_code_sums` varies a bit
                    # depending on torch version
                    self.assertTrue(
                        np.abs(
                            audio_code_sums - expected_codesums[num_codebooks]) <= (3e-3 * audio_code_sums)
                    )

                    input_values_dec = model.decode(
                        encoder_outputs[0], padding_mask=inputs["padding_mask"])[0]
                    input_values_enc_dec = model(
                        inputs["input_values"], inputs["padding_mask"], num_quantizers=int(
                            num_codebooks)
                    )[1]

                # make sure forward and decode gives same result
                self.assertTrue(
                    allclose(input_values_dec, input_values_enc_dec))

                # make sure shape matches
                self.assertTrue(
                    inputs["input_values"].shape == input_values_enc_dec.shape)

                arr = inputs["input_values"][0].numpy()
                arr_enc_dec = input_values_enc_dec[0].numpy()

                # make sure audios are more or less equal
                # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
                rmse = compute_rmse(arr, arr_enc_dec)
                self.assertTrue(np.abs(rmse - expected_rmse) < 1e-5)
