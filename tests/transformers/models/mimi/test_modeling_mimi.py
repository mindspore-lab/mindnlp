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
"""Testing suite for the PyTorch Mimi model."""

import inspect
import os
import tempfile
import unittest

import sys
import numpy as np
from datasets import Audio, load_dataset
from mindnlp.utils.testing_utils import require_mindspore
from parameterized import parameterized
from pytest import mark
from mindspore import ops,nn

from mindnlp.transformers import AutoFeatureExtractor, MimiConfig

from mindnlp.utils.testing_utils import (
    is_mindspore_available,
    is_flaky,
    # require_accelerate,
    # require_bitsandbytes,
    # require_flash_attn,
    # require_read_token,
    require_safetensors,
    require_mindspore,
    slow,
)

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,parentdir)

from tests.transformers.test_configuration_common import ConfigTester
from tests.transformers.test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from tests.transformers.generation.test_utils import (
    is_flaky,
    is_mindspore_available,
    slow
)

if is_mindspore_available():
    import mindspore

    from mindspore import ops
    from mindnlp.transformers.models  import MimiModel


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

    decoder_dict = {"decoder_input_ids": decoder_input_ids} if decoder_input_ids is not None else {}

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
        input_values = floats_tensor([self.batch_size, self.num_channels, self.intermediate_size], scale=1.0)
        config = self.get_config()
        inputs_dict = {"input_values": input_values}
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
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
        model = MimiModel(config=config)
        model.set_train(False)

        input_values = inputs_dict["input_values"]
        result = model(input_values)
        self.parent.assertEqual(
            result.audio_values.shape, (self.batch_size, self.num_channels, self.intermediate_size)
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
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
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

            expected_arg_names = ["input_values", "padding_mask", "num_quantizers"]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

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


    @unittest.skip(reason="The MimiModel does not have the usual `attention` logic")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The MimiModel does not have the usual `hidden_states` logic")
    def test_hidden_states_output(self):
        pass

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_determinism
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

            model.set_train(False)

            first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
            second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_model_outputs_equivalence
    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            
            tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs)

            self.assertTrue(isinstance(tuple_output, tuple))
            self.assertTrue(isinstance(dict_output, dict))

            for tuple_value, dict_value in zip(tuple_output, dict_output.values()):
                a = set_nan_tensor_to_zero(tuple_value)
                b = set_nan_tensor_to_zero(dict_value)
                a = a.asnumpy()
                b = b.asnumpy()
                w,h,c = a.shape
                test = True
                for i in range(w):
                    for j in range(h):
                        for k in range(c):
                            if(abs(a[i,j,k]-b[i,j,k])>0.00001):
                                print("Tuple and dict output are not equal. Difference:")
                                print(a[i,j,k])
                                test = False
                #"Tuple and dict output are not equal. Difference:"+
                # msg=(
                #         f"Difference: {ops.max(ops.abs(tuple_value - dict_value))}. Tuple has `nan`:"+f" {ops.isnan(tuple_value).any()} and `inf`: {ops.isinf(tuple_value)}. Dict has"+f" `nan`: {ops.isnan(dict_value).any()} and `inf`: {ops.isinf(dict_value)}."
                #     ),
                self.assertTrue(
                    test==True
                )

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.set_train(False)

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
                            -1.0 <= ((param.data.mean() * 1e9).round() / 1e9).item() <= 1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    # Copied from transformers.tests.encodec.test_modeling_encodec.MimiModelTest.test_identity_shortcut
    def test_identity_shortcut(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs()
        config.use_conv_shortcut = False
        self.model_tester.create_and_check_model_forward(config, inputs_dict)

    # Overwrite to use `audio_values` as the tensors to compare.
    # TODO: Try to do this in the parent class.
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_mindspore
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        self.skipTest(reason="Idefics has a hard requirement on SDPA, skipping this test")
        
        # torch_device = mindspore.get_context('device_target')
        # if torch_dtype == "float16" and torch_device == "cpu":
        #     self.skipTest("`replication_pad1d` not implemented for 'Half")

        # if not self.has_attentions:
        #     self.skipTest(reason="Model architecture does not support attentions")

        # if not self.all_model_classes[0]._supports_sdpa:
        #     self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")


        # # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        # if torch_dtype == "float16":
        #     torch_dtype = mindspore.float16
        # elif torch_dtype == "bfloat16":
        #     torch_dtype = mindspore.bfloat16
        # elif torch_dtype == "float32":
        #     torch_dtype = mindspore.float32

        # atols = {
        #     ("cpu", False, mindspore.float32): 1e-6,
        #     ("cpu", False, mindspore.bfloat16): 1e-2,
        #     ("cpu", True, mindspore.float32): 1e-6,
        #     ("cpu", True, mindspore.bfloat16): 1e-2,
        #     ("Ascend", False, mindspore.float32): 1e-6,
        #     ("Ascend", False, mindspore.bfloat16): 1e-2,
        #     ("Ascend", False, mindspore.float16): 5e-3,
        #     ("Ascend", True, mindspore.float32): 1e-6,
        #     ("Ascend", True, mindspore.bfloat16): 1e-2,
        #     ("Ascend", True, mindspore.float16): 5e-3,
        # }
        # rtols = {
        #     ("cpu", False, mindspore.float32): 1e-4,
        #     ("cpu", False, mindspore.bfloat16): 1e-2,
        #     ("cpu", True, mindspore.float32): 1e-4,
        #     ("cpu", True, mindspore.bfloat16): 1e-2,
        #     ("Ascend", False, mindspore.float32): 1e-4,
        #     ("Ascend", False, mindspore.bfloat16): 1e-2,
        #     ("Ascend", False, mindspore.float16): 5e-3,
        #     ("Ascend", True, mindspore.float32): 1e-4,
        #     ("Ascend", True, mindspore.bfloat16): 3e-2,
        #     ("Ascend", True, mindspore.float16): 5e-3,
        # }

        # def get_mean_reldiff(failcase, x, ref, atol, rtol):
        #     return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, Ascend atol = {atol}, Ascend rtol = {rtol}"

        # for model_class in self.all_model_classes:
        #     config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        #     model = model_class(config)
        #     # FIXME: we deactivate boolean mask for models using "use_mask_token" in their constructors.
        #     # These models support masking only in the case `use_mask_token=True`. Otherwise they cannot consume an input mask.
        #     # This means that the class needs to be instantiated much later, after `use_mask` is set, which means a significant refactor of the code.
        #     # However masking there is not done at any layers that matters (i.e self-attention), therefore we can safely deactivate it.
        #     deactivate_mask = "use_mask_token" in inspect.signature(model_class).parameters

        #     is_encoder_decoder = model.config.is_encoder_decoder

        #     with tempfile.TemporaryDirectory() as tmpdirname:
        #         model.save_pretrained(tmpdirname)
        #         model_sdpa = model_class.from_pretrained(tmpdirname)
        #         model_sdpa = model_sdpa.set_train(False)

        #         self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

        #         model_eager = model_class.from_pretrained(
        #             tmpdirname,
        #             attn_implementation="eager",
        #         )
        #         model_eager = model_eager.set_train(False)

        #         self.assertTrue(model_eager.config._attn_implementation == "eager")

        #         for name, submodule in model_eager.named_modules():
        #             class_name = submodule.__class__.__name__
        #             if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
        #                 raise ValueError("The eager model should not have SDPA attention layers")

        #         has_sdpa = False
        #         for name, submodule in model_sdpa.named_modules():
        #             class_name = submodule.__class__.__name__
        #             if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
        #                 has_sdpa = True
        #                 break
        #         if not has_sdpa and model_sdpa.config.model_type != "falcon":
        #             raise ValueError("The SDPA model should have SDPA attention layers")

        #         # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 16 times the model,
        #         # but it would be nicer to have an efficient way to use parameterized.expand
        #         fail_cases = []
        #         for padding_side in ["left", "right"]:
        #             for use_mask in [False, True]:
        #                 for output_attentions in [True, False]:
        #                     can_output_attn = "output_attentions" in inspect.signature(model_sdpa.forward).parameters
        #                     if not (self.has_attentions and can_output_attn) and output_attentions:
        #                         continue
        #                     for batch_size in [7]:
        #                         dummy_input = inputs_dict[model.main_input_name]

        #                         if dummy_input.dtype in [mindspore.float32, mindspore.bfloat16, mindspore.float16]:
        #                             dummy_input = dummy_input

        #                         dummy_input = dummy_input[:batch_size]
        #                         if dummy_input.shape[0] != batch_size:
        #                             if dummy_input.dtype in [mindspore.float32, mindspore.bfloat16, mindspore.float16]:
        #                                 extension = ops.rand(
        #                                     batch_size - dummy_input.shape[0],
        #                                     *dummy_input.shape[1:]
        #                                 )
        #                                 dummy_input = ops.cat((dummy_input, extension), axis=0)
        #                             else:
        #                                 extension = ops.randint(
        #                                     high=5,
        #                                     size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
        #                                     dtype=dummy_input.dtype
        #                                 )
        #                                 dummy_input = ops.cat((dummy_input, extension), axis=0)

        #                         if not use_mask:
        #                             dummy_attention_mask = None
        #                         else:
        #                             dummy_attention_mask = inputs_dict.get("attention_mask", None)
        #                             if dummy_attention_mask is None:
        #                                 if is_encoder_decoder:
        #                                     seqlen = inputs_dict.get("decoder_input_ids", dummy_input).shape[-1]
        #                                 else:
        #                                     seqlen = dummy_input.shape[-1]
        #                                 dummy_attention_mask = (
        #                                     ops.ones((batch_size, seqlen), dtype=mindspore.int64)
        #                                 )

        #                             dummy_attention_mask = dummy_attention_mask[:batch_size]
        #                             if dummy_attention_mask.shape[0] != batch_size:
        #                                 extension = ops.ones(
        #                                     (batch_size - dummy_attention_mask.shape[0],
        #                                     *dummy_attention_mask.shape[1:]),
        #                                     dtype=dummy_attention_mask.dtype
        #                                 )
        #                                 dummy_attention_mask = ops.cat((dummy_attention_mask, extension), axis=0)
        #                                 dummy_attention_mask = dummy_attention_mask.to(torch_device)

        #                             dummy_attention_mask[:] = 1
        #                             if padding_side == "left":
        #                                 dummy_attention_mask[-1, :2] = 0
        #                                 dummy_attention_mask[-1, 2:] = 1
        #                             elif padding_side == "right":
        #                                 dummy_attention_mask[-1, -2:] = 0
        #                                 dummy_attention_mask[-1, :-2] = 1

        #                         for enable_kernels in [False, True]:
        #                             failcase = f"padding_side={padding_side}, use_mask={use_mask}, batch_size={batch_size}, enable_kernels={enable_kernels}"
        #                             if is_encoder_decoder:
        #                                 decoder_input_ids = inputs_dict.get("decoder_input_ids", dummy_input)[
        #                                     :batch_size
        #                                 ]
        #                                 if decoder_input_ids.shape[0] != batch_size:
        #                                     extension = ops.ones((
        #                                         batch_size - decoder_input_ids.shape[0],
        #                                         *decoder_input_ids.shape[1:]),
        #                                         dtype=decoder_input_ids.dtype
        #                                     )
        #                                     decoder_input_ids = ops.cat((decoder_input_ids, extension), axis=0)
        #                                     decoder_input_ids = decoder_input_ids

        #                                 # TODO: never an `attention_mask` arg here?
        #                                 processed_inputs = {
        #                                     model.main_input_name: dummy_input,
        #                                     "decoder_input_ids": decoder_input_ids,
        #                                     "decoder_attention_mask": dummy_attention_mask,
        #                                     "output_hidden_states": True,
        #                                 }
        #                             else:
        #                                 processed_inputs = {
        #                                     model.main_input_name: dummy_input,
        #                                     "output_hidden_states": True,
        #                                 }

        #                                 # Otherwise fails for e.g. WhisperEncoderModel
        #                                 if "attention_mask" in inspect.signature(model_eager.forward).parameters:
        #                                     processed_inputs["attention_mask"] = dummy_attention_mask

        #                                 if (
        #                                     self.has_attentions
        #                                     and "output_attentions" in inspect.signature(model_sdpa.forward).parameters
        #                                 ):
        #                                     processed_inputs["output_attentions"] = output_attentions
        #                             if not deactivate_mask and (
        #                                 "bool_masked_pos" in inspect.signature(model_eager.forward).parameters
        #                             ):
        #                                 dummy_mask = ops.ones((self.model_tester.num_masks,))

        #                                 # In case of additional token (like class) we define a custom `mask_length`
        #                                 if hasattr(self.model_tester, "mask_length"):
        #                                     mask_length = self.model_tester.mask_length - dummy_mask.size(0)
        #                                 else:
        #                                     mask_length = self.model_tester.seq_length - dummy_mask.size(0)
        #                                 dummy_mask = ops.cat([dummy_mask, ops.zeros(mask_length)])
        #                                 dummy_bool_masked_pos = dummy_mask.expand(batch_size, -1).bool()
        #                                 processed_inputs["bool_masked_pos"] = dummy_bool_masked_pos.to(torch_device)

        #                             if "noise" in inspect.signature(model_eager.forward).parameters:
        #                                 np.random.seed(2)
        #                                 num_patches = int(
        #                                     (self.model_tester.image_size // self.model_tester.patch_size) ** 2
        #                                 )
        #                                 noise = np.random.uniform(size=(batch_size, num_patches))
        #                                 processed_inputs["noise"] = mindspore.from_numpy(noise)

        #                             # TODO: test gradients as well (& for FA2 as well!)

        #                             with sdpa_kernel(
        #                                 enable_flash=enable_kernels,
        #                                 enable_math=True,
        #                                 enable_mem_efficient=enable_kernels,
        #                             ):
        #                                 prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
        #                                 outputs_eager = model_eager(**prepared_inputs)
        #                                 outputs_sdpa = model_sdpa(**prepared_inputs)

        #                             # Ignore copy
        #                             logits_eager = outputs_eager.audio_values
        #                             # Ignore copy
        #                             logits_sdpa = outputs_sdpa.audio_values

        #                             if torch_device in ["cpu", "Ascend"]:
        #                                 atol = atols[torch_device, enable_kernels, torch_dtype]
        #                                 rtol = rtols[torch_device, enable_kernels, torch_dtype]
        #                             elif torch_device == "xpu":
        #                                 # As of PyTorch 2.5 XPU backend supports only torch.nn.attention.SDPBackend.MATH
        #                                 # which is implemented on PyTorch level using aten operators and is
        #                                 # device agnostic with respect to implementation of each aten operator.
        #                                 atol = atols["Ascend", False, torch_dtype]
        #                                 rtol = rtols["Ascend", False, torch_dtype]
        #                             else:
        #                                 atol = 1e-7
        #                                 rtol = 1e-4

        #                             # Masked tokens output slightly deviates - we don't mind that.
        #                             if use_mask:
        #                                 _logits_sdpa = ops.zeros_like(input=logits_sdpa)
        #                                 _logits_eager = ops.zeros_like(input=logits_eager)

        #                                 _logits_sdpa[:-1] = logits_sdpa[:-1]
        #                                 _logits_eager[:-1] = logits_eager[:-1]

        #                                 if padding_side == "left":
        #                                     _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
        #                                     _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

        #                                 elif padding_side == "right":
        #                                     _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
        #                                     _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

        #                                 logits_sdpa = _logits_sdpa
        #                                 logits_eager = _logits_eager

        #                             results = [
        #                                 np.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
        #                                 for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
        #                             ]
        #                             # If 80% batch elements have matched results, it's fine
        #                             if np.mean(results) < 0.8:
        #                                 fail_cases.append(
        #                                     get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
        #                                 )

        #         self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))


    @require_mindspore
    @slow
    @is_flaky()
    def test_flash_attn_2_inference_equivalence(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname,  attn_implementation="flash_attention_2"
                )

                model = model_class.from_pretrained(tmpdirname)

                dummy_input = inputs_dict[model.main_input_name][:1]
                if dummy_input.dtype in [mindspore.float32, mindspore.float16]:
                    dummy_input = dummy_input.to(mindspore.bfloat16)

                outputs = model(dummy_input)
                outputs_fa = model_fa(dummy_input)

                logits = outputs[1]
                logits_fa = outputs_fa[1]

                assert np.allclose(logits_fa, logits, atol=4e-2, rtol=4e-2)

    @unittest.skip(reason="The MimiModel does not support right padding")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="The MimiModel does not have support dynamic compile yet")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @is_flaky()
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
        expected_rmse = {
            "8": 0.0018785292,
            "32": 0.0012330565,
        }

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        model = MimiModel.from_pretrained(model_id, use_cache=True)
        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )
        model.set_train(False)
        for num_codebooks, expected_rmse in expected_rmse.items():
            # use max bandwith for best possible reconstruction
            encoder_outputs = model.encode(inputs["input_values"], num_quantizers=int(num_codebooks))

            audio_codes = encoder_outputs[0]

            decoder_outputs_first_part = model.decode(audio_codes[:, :, : audio_codes.shape[2] // 2])
            decoder_outputs_second_part = model.decode(
                audio_codes[:, :, audio_codes.shape[2] // 2 :],
                decoder_past_key_values=decoder_outputs_first_part.decoder_past_key_values,
            )

            audio_output_entire_context = model.decode(audio_codes)[0]
            audio_output_concat_context = ops.cat(
                [decoder_outputs_first_part[0], decoder_outputs_second_part[0]], axis=2
            )

            # make sure audios are more or less equal
            # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
            rmse = compute_rmse(
                audio_output_concat_context.squeeze().numpy(),
                audio_output_entire_context.squeeze().numpy(),
            )
            self.assertTrue(rmse < 1e-3)

    def test_integration(self):
        expected_rmses = {
            "8": 0.0018785292,
            "32": 0.0012330565,
        }
        expected_codesums = {
            "8": 430423,
            "32": 1803071,
        }
        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        model_id = "kyutai/mimi"

        processor = AutoFeatureExtractor.from_pretrained(model_id)

        librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
        audio_sample = librispeech_dummy[-1]["audio"]["array"]

        inputs = processor(
            raw_audio=audio_sample,
            sampling_rate=processor.sampling_rate,
            return_tensors="pt",
        )

        for use_cache in [False, True]:
            model = MimiModel.from_pretrained(model_id, use_cache=use_cache)
            model.set_train(False)
            for num_codebooks, expected_rmse in expected_rmses.items():
                # use max bandwith for best possible reconstruction
                encoder_outputs = model.encode(inputs["input_values"], num_quantizers=int(num_codebooks))

                audio_code_sums = encoder_outputs[0].sum().item()

                # make sure audio encoded codes are correct
                # assert relative difference less than a threshold, because `audio_code_sums` varies a bit
                # depending on torch version
                self.assertTrue(
                    np.abs(audio_code_sums - expected_codesums[num_codebooks]) <= (3e-3 * audio_code_sums)
                )

                input_values_dec = model.decode(encoder_outputs[0], padding_mask=inputs["padding_mask"])[0]
                input_values_enc_dec = model(
                    inputs["input_values"], inputs["padding_mask"], num_quantizers=int(num_codebooks)
                )[1]

                # make sure forward and decode gives same result
                self.assertTrue(np.allclose(input_values_dec, input_values_enc_dec))

                # make sure shape matches
                self.assertTrue(inputs["input_values"].shape == input_values_enc_dec.shape)

                arr = inputs["input_values"][0].numpy()
                arr_enc_dec = input_values_enc_dec[0].numpy()

                # make sure audios are more or less equal
                # the RMSE of two random gaussian noise vectors with ~N(0, 1) is around 1.0
                rmse = compute_rmse(arr, arr_enc_dec)
                self.assertTrue(np.abs(rmse - expected_rmse) < 1e-5)
