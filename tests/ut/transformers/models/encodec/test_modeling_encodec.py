""" Testing suite for the MindSpore Encodec model. """

import copy
import inspect
import unittest
from typing import Dict, List, Tuple
import numpy as np


from mindnlp.utils.testing_utils import (
    is_mindspore_available,
    require_mindspore,
    slow,
)
from mindnlp.transformers import EncodecModel, EncodecConfig
from mindnlp.transformers.models.encodec import EncodecFeatureExtractor
from datasets import Audio, load_dataset

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
)


# from ...test_pipeline_mixin import PipelineTesterMixin

if is_mindspore_available():
    import mindspore
    from mindnlp.core import ops


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
    r"""
    prepare inputs
    """
    if input_ids is not None:
        encoder_dict = {"input_ids": input_ids}
    else:
        encoder_dict = {"input_values": input_values}

    decoder_dict = {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}

    return {**encoder_dict, **decoder_dict}


@require_mindspore
class EncodecModelTester:
    r"""
    EncodecModel Tester
    """
    def __init__(
        self,
        parent,
        # `batch_size` needs to be an even number if the model has some outputs with batch dim != 0.
        batch_size=12,
        num_channels=2,
        is_training=False,
        intermediate_size=40,
        hidden_size=32,
        num_filters=8,
        num_residual_layers=1,
        upsampling_ratios=[8, 4],
        num_lstm_layers=1,
        codebook_size=64,
    ):
        r"""
        Test EncodecModel
        """
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_residual_layers = num_residual_layers
        self.upsampling_ratios = upsampling_ratios
        self.num_lstm_layers = num_lstm_layers
        self.codebook_size = codebook_size

    def prepare_config_and_inputs(self):
        r"""
        prepare config and inputs
        """
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        r"""
        prepare config and inputs for common
        """
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_config(self):
        r"""
        obtain EncodecModel Config
        """
        return EncodecConfig(
            audio_channels=self.num_channels,
            chunk_in_sec=None,
            hidden_size=self.hidden_size,
            num_filters=self.num_filters,
            num_residual_layers=self.num_residual_layers,
            upsampling_ratios=self.upsampling_ratios,
            num_lstm_layers=self.num_lstm_layers,
            codebook_size=self.codebook_size,
        )

    def create_and_check_model_forward(self, config, inputs_dict):
        r"""
        Encodec Model check method
        """
        model = EncodecModel(config=config).set_train(False)

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size, self.num_channels, self.intermediate_size)
        )


@require_mindspore
class EncodecModelTest(ModelTesterMixin, unittest.TestCase):
    r"""
    EncodecModel Test
    """
    all_model_classes = (EncodecModel,) if is_mindspore_available() else ()
    is_encoder_decoder = True
    test_pruning = False
    test_headmasking = False
    test_resize_embeddings = False
    pipeline_model_mapping = {"feature-extraction": EncodecModel} if is_mindspore_available() else {}
    input_name = "input_values"

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # model does not have attention and does not support returning hidden states
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "output_attentions" in inputs_dict:
            inputs_dict.pop("output_attentions")
        if "output_hidden_states" in inputs_dict:
            inputs_dict.pop("output_hidden_states")
        return inputs_dict

    def setUp(self):
        self.model_tester = EncodecModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=EncodecConfig, hidden_size=37, common_properties=[], has_text_modality=False
        )

    def test_config(self):
        r"""
        test config
        """
        self.config_tester.run_common_tests()

    def test_model_forward(self):
        r"""
        test_model_forward
        """
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_values", "padding_mask", "bandwidth"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    @unittest.skip("The EncodecModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("The EncodecModel is not transformers based, thus it does not have `inputs_embeds` logics")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("The EncodecModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_retain_grad_hidden_states_attentions(self):
        r""" 
        skip 
        """

    @unittest.skip("The EncodecModel is not transformers based, thus it does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    def test_feed_forward_chunking(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            config.chunk_length_s = None
            config.overlap = None
            config.sampling_rate = 10

            model = model_class(config)
            model.set_train(False)
            inputs = self._prepare_for_class(inputs_dict, model_class)
            inputs["input_values"] = ops.tile(inputs["input_values"], (1, 1, 10))

            hidden_states_no_chunk = model(**inputs)[0]

            config.chunk_length_s = 1
            config.overlap = 0
            config.sampling_rate = 10

            model = model_class(config)
            model.set_train(False)

            hidden_states_with_chunk = model(**inputs)[0]
            # print(hidden_states_no_chunk)
            self.assertTrue(np.allclose(hidden_states_no_chunk.asnumpy().astype(float), hidden_states_with_chunk.asnumpy().astype(float), atol=1e-3))

    @unittest.skip("The EncodecModel is not transformers based, thus it does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # outputs are not tensors but list (since each sequence don't have the same frame_length)
            out_1 = first.asnumpy()
            out_2 = second.asnumpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.set_train(False)
            first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t
        #pylint: disable=W0102
        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            r"""
            check_equivalence
            """
            tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs)

            def recursive_check(tuple_object, dict_object):
                if isinstance(tuple_object, (List, Tuple)):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif isinstance(tuple_object, Dict):
                    for tuple_iterable_value, dict_iterable_value in zip(
                        tuple_object.values(), dict_object.values()
                    ):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assertTrue(
                        np.allclose(
                            set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                        ),
                        msg=(
                            "Tuple and dict output are not equal. Difference:"
                            f" {ops.max(ops.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                            f" {ops.isnan(tuple_object).any()} and `inf`: {ops.isinf(tuple_object)}. Dict has"
                            f" `nan`: {ops.isnan(dict_object).any()} and `inf`: {ops.isinf(dict_object)}."
                        ),
                    )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.set_train(False)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.parameters_and_names():
                uniform_init_parms = ["conv"]
                ignore_init = ["lstm"]
                if param.requires_grad:
                    if any(x in name for x in uniform_init_parms):
                        self.assertTrue(
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    elif not any(x in name for x in ignore_init):
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_identity_shortcut(self):
        r"""
        test_identity_shortcut
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)


def normalize(arr):
    r"""
    norm method
    """
    norm = np.linalg.norm(arr)
    normalized_arr = arr / norm
    return normalized_arr


def compute_rmse(arr1, arr2):
    r"""
    compute rmse
    """
    arr1_normalized = normalize(arr1)
    arr2_normalized = normalize(arr2)
    return np.sqrt(((arr1_normalized - arr2_normalized) ** 2).mean())


@slow
@require_mindspore
class EncodecIntegrationTest(unittest.TestCase):
    r"""
    Test Encodec Integration
    """
    def test_integration_24kHz(self):
        r"""
        24KHz
        """
        expected_rmse = {
            "1.5": 0.0025,
            "24.0": 0.0015,
        }
        expected_codesums = {
            "1.5": [371955],
            "24.0": [6659962],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "facebook/encodec_24khz"

        model = EncodecModel.from_pretrained(model_id)
        processor = EncodecFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="ms",
        )

        for bandwidth, expected_rmse in expected_rmse.items():

            # use max bandwith for best possible reconstruction
            encoder_outputs = model.encode(inputs["input_values"], bandwidth=float(bandwidth))

            audio_code_sums = [a[0].sum().item() for a in encoder_outputs[0]]

            # make sure audio encoded codes are correct
            self.assertListEqual(audio_code_sums, expected_codesums[bandwidth])

            audio_codes, scales = encoder_outputs.to_tuple()
            input_values_dec = model.decode(audio_codes, scales, inputs["padding_mask"])[0]
            input_values_enc_dec = model(
                inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth)
            )[-1]

            # make sure forward and decode gives same result
            self.assertTrue(np.allclose(input_values_dec.asnumpy(), input_values_enc_dec.asnumpy(), atol=1e-3))

            # make sure shape matches
            self.assertTrue(inputs["input_values"].shape == input_values_enc_dec.shape)

            arr = inputs["input_values"][0].asnumpy()
            arr_enc_dec = input_values_enc_dec[0].asnumpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            print(rmse, expected_rmse)
            self.assertTrue(rmse < expected_rmse)

    def test_integration_48kHz(self):
        r"""
        48KHz test
        """
        expected_rmse = {
            "3.0": 0.001,
            "24.0": 0.0005,
        }
        expected_codesums = {
            "3.0": [144259, 146765, 156435, 176871, 161971],
            "24.0": [1568553, 1294948, 1306190, 1464747, 1663150],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "facebook/encodec_48khz"

        model = EncodecModel.from_pretrained(model_id)
        model.set_train(False)
        processor = EncodecFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        # transform mono to stereo
        audio_sample = np.array([audio_sample, audio_sample])

        inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="ms")

        for bandwidth, expected_rmse in expected_rmse.items():

            # use max bandwith for best possible reconstruction
            encoder_outputs = model.encode(
                inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth), return_dict=False
            )
            audio_code_sums = [a[0].sum().item() for a in encoder_outputs[0]]

            # make sure audio encoded codes are correct
            self.assertListEqual(audio_code_sums, expected_codesums[bandwidth])
            audio_codes, scales = encoder_outputs
            input_values_dec = model.decode(audio_codes, scales, inputs["padding_mask"])[0]
            input_values_enc_dec = model(
                inputs["input_values"], inputs["padding_mask"], bandwidth=float(bandwidth)
            )[-1]

            # make sure forward and decode gives same result
            self.assertTrue(np.allclose(input_values_dec.asnumpy(), input_values_enc_dec.asnumpy(), atol=1e-3))

            # make sure shape matches
            self.assertTrue(inputs["input_values"].shape == input_values_enc_dec.shape)

            arr = inputs["input_values"][0].asnumpy()
            arr_enc_dec = input_values_enc_dec[0].asnumpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)

    def test_batch_48kHz(self):
        r"""
        batch_48khz
        """
        expected_rmse = {
            "3.0": 0.001,
            "24.0": 0.0005,
        }
        expected_codesums = {
            "3.0": [
                [72410, 79137, 76694, 90854, 73023, 82980, 72707, 54842],
                [85561, 81870, 76953, 48967, 79315, 85442, 81479, 107241],
            ],
            "24.0": [
                [72410, 79137, 76694, 90854, 73023, 82980, 72707, 54842],
                [85561, 81870, 76953, 48967, 79315, 85442, 81479, 107241],
            ],
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        model_id = "facebook/encodec_48khz"

        model = EncodecModel.from_pretrained(model_id)
        processor = EncodecFeatureExtractor.from_pretrained(model_id, chunk_length_s=1, overlap=0.01)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

        audio_samples = [
            np.array([audio_sample["array"], audio_sample["array"]])
            for audio_sample in librispeech_dummy[-2:]["audio"]
        ]

        inputs = processor(raw_audio=audio_samples, sampling_rate=processor.sampling_rate, return_tensors="ms")
        input_values = inputs["input_values"]
        for bandwidth, expected_rmse in expected_rmse.items():
            # use max bandwith for best possible reconstruction
            encoder_outputs = model.encode(input_values, bandwidth=float(bandwidth), return_dict=False)
            audio_code_sums_0 = [a[0][0].sum().item() for a in encoder_outputs[0]]
            audio_code_sums_1 = [a[0][1].sum().item() for a in encoder_outputs[0]]

            # make sure audio encoded codes are correct
            self.assertListEqual(audio_code_sums_0, expected_codesums[bandwidth][0])
            self.assertListEqual(audio_code_sums_1, expected_codesums[bandwidth][1])

            audio_codes, scales = encoder_outputs
            input_values_dec = model.decode(audio_codes, scales)[0]
            input_values_enc_dec = model(input_values, bandwidth=float(bandwidth))[-1]

            # make sure forward and decode gives same result
            self.assertTrue(np.allclose(input_values_dec, input_values_enc_dec, atol=1e-3))

            # make sure shape matches
            self.assertTrue(input_values.shape == input_values_enc_dec.shape)

            arr = input_values[0].asnumpy()
            arr_enc_dec = input_values_enc_dec[0].asnumpy()

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(arr, arr_enc_dec)
            self.assertTrue(rmse < expected_rmse)
