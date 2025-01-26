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
"""Testing suite for the PyTorch SigLIP model."""

import inspect
import os
import tempfile
import unittest
from typing import Tuple

import numpy as np
import requests
from parameterized import parameterized
from pytest import mark

from mindnlp.utils import (
    is_mindspore_available,
    is_vision_available,
)
from mindnlp.utils.testing_utils import (
    require_mindspore,
    require_mindspore_gpu,
    require_vision,
    slow,
)
from mindnlp.transformers.models.siglip.configuration_siglip import (
    SiglipConfig,
    SiglipTextConfig,
    SiglipVisionConfig
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    is_flaky,
    random_attention_mask,
)
# from ...test_pipeline_mixin import PipelineTesterMixin


if is_mindspore_available():
    import mindspore
    from mindnlp.core import nn, ops

    from mindnlp.transformers.models.siglip.modeling_siglip import (
        SiglipForImageClassification,
        SiglipModel,
        SiglipTextModel,
        SiglipVisionModel
    )

if is_vision_available():
    from PIL import Image

    from mindnlp.transformers.models.siglip.processing_siglip import SiglipProcessor


class SiglipModelTesterMixin(ModelTesterMixin):
    def test_sdpa_can_dispatch_composite_models(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.set_train(False)

            if hasattr(model_eager, "vision_model") and hasattr(model_eager, "text_model"):
                self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.text_model.config._attn_implementation == "eager")

            self.assertTrue(model_eager.config._attn_implementation == "eager")

            for name, submodule in model_eager.named_modules():
                class_name = submodule.__class__.__name__
                if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                    raise ValueError("The eager model should not have SDPA attention layers")

    # def test_sdpa_can_dispatch_composite_models(self):
    #     for model_class in self.all_model_classes:
    #         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #         model = model_class(config)

    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             model.save_pretrained(tmpdirname)

    #             # Load the model with SDPA
    #             model_sdpa = model_class.from_pretrained(tmpdirname,attn_implementation="sdpa",)
    #             model_sdpa = model_sdpa.set_train(False)

    #             # Load model with eager attention
    #             model_eager = model_class.from_pretrained(
    #                 tmpdirname,
    #                 attn_implementation="eager",
    #             )
    #             model_eager = model_eager.set_train(False)

    #         # SigLip has one shared cls attr for all models, so we assign both submodels heer
    #         vision_attn = text_attn = "sdpa" if model._supports_sdpa else "eager"

    #         if hasattr(model_sdpa, "vision_model") and hasattr(model_sdpa, "text_model"):
    #             self.assertTrue(model_sdpa.vision_model.config._attn_implementation == vision_attn)
    #             self.assertTrue(model_sdpa.text_model.config._attn_implementation == text_attn)
    #             self.assertTrue(model_eager.vision_model.config._attn_implementation == "eager")
    #             self.assertTrue(model_eager.text_model.config._attn_implementation == "eager")

    #         self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
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

    def test_eager_matches_sdpa_inference(
        self,
        ms_dtype: str,
        use_attention_mask_options: Tuple[bool,...] = (True, False),
        logit_keys: Tuple[str,...] = ("logits_per_image", "logits_per_text", "image_embeds", "text_embeds"),
    ):
        # Convert to torch dtype
        dtypes = {
            "float16": mindspore.float16,
            "float32": mindspore.float32,
        }
        ms_dtype = dtypes[ms_dtype]

        atols = {
            mindspore.float32: 1e-5,
            mindspore.float16: 5e-3,
        }
        rtols = {
            mindspore.float32: 1e-4,
            mindspore.float16: 5e-3,
        }

        atol = atols[ms_dtype]
        rtol = rtols[ms_dtype]

        def get_mean_reldiff(msg, current_case, x, ref, atol, rtol):
            return f"{msg} {current_case}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Load the model with eager attention
                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    ms_dtype=ms_dtype,
                    attn_implementation="eager",
                )
                model_eager = model_eager.set_train(False)

            cases = [
                (use_mask, output_attentions, batch_size)
                for use_mask in use_attention_mask_options
                for output_attentions in [True, False]
                for batch_size in [1, 5]
            ]
            fail_cases = []

            for use_mask, output_attentions, batch_size in cases:
                processed_inputs = inputs_dict.copy()

                # convert to ms_dtype
                if "pixel_values" in processed_inputs:
                    processed_inputs["pixel_values"] = processed_inputs["pixel_values"].to(dtype=ms_dtype)

                # slice for different batch sizes
                for key in ["pixel_values", "input_ids", "attention_mask"]:
                    if key in processed_inputs:
                        processed_inputs[key] = processed_inputs[key][:batch_size]

                # set attention mask with left padding
                if not use_mask:
                    processed_inputs.pop("attention_mask", None)
                else:
                    dummy_attention_mask = processed_inputs["attention_mask"]
                    dummy_attention_mask[:] = 1
                    processed_inputs["attention_mask"] = dummy_attention_mask

                processed_inputs["output_attentions"] = output_attentions
                processed_inputs["output_hidden_states"] = True

                current_case = (
                    f"padding_side=left, use_mask={use_mask}, batch_size={batch_size}"
                )

                prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                
                try:
                    outputs_eager = model_eager(**prepared_inputs)
                except Exception as e:
                    fail_cases.append(f"{current_case}: {e}")
                    continue

                for key in logit_keys:
                    eager_logits = outputs_eager[key]

                    if use_mask:
                        eager_logits = eager_logits[:, 1:]

            self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))

    # def test_eager_matches_sdpa_inference(
    #     self,
    #     ms_dtype: str,
    #     use_attention_mask_options: Tuple[bool, ...] = (True, False),
    #     logit_keys: Tuple[str, ...] = ("logits_per_image", "logits_per_text", "image_embeds", "text_embeds"),
    # ):
    #     if not self.all_model_classes[0]._supports_sdpa:
    #         self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

    #     # Convert to torch dtype
    #     dtypes = {
    #         "float16": mindspore.float16,
    #         "bfloat16": mindspore.bfloat16,
    #         "float32": mindspore.float32,
    #     }
    #     ms_dtype = dtypes[ms_dtype]

    #     atols = {
    #         mindspore.float32: 1e-5,
    #         mindspore.bfloat16: 3e-2,
    #         mindspore.float16: 5e-3,
    #     }
    #     rtols = {
    #         mindspore.float32: 1e-4,
    #         mindspore.bfloat16: 3e-2,
    #         mindspore.float16: 5e-3,
    #     }

    #     atol = atols[ms_dtype]
    #     rtol = rtols[ms_dtype]

    #     def get_mean_reldiff(msg, current_case, x, ref, atol, rtol):
    #         return f"{msg} {current_case}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

    #     for model_class in self.all_model_classes:
    #         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    #         model = model_class(config)

    #         with tempfile.TemporaryDirectory() as tmpdirname:
    #             model.save_pretrained(tmpdirname)

    #             # Load the model with SDPA
    #             model_sdpa = model_class.from_pretrained(tmpdirname, ms_dtype=ms_dtype,attn_implementation="sdpa")
    #             model_sdpa = model_sdpa.set_train(False)

    #             # Load model with eager attention
    #             model_eager = model_class.from_pretrained(
    #                 tmpdirname,
    #                 ms_dtype=ms_dtype,
    #                 attn_implementation="eager",
    #             )
    #             model_eager = model_eager.set_train(False)

    #         # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving the model each time,
    #         # but it would be nicer to have an efficient way to use parameterized.expand
    #         cases = [
    #             (use_mask, output_attentions, sdpa_backend, batch_size)
    #             for use_mask in use_attention_mask_options
    #             for output_attentions in [True, False]
    #             for sdpa_backend in [
    #                 SDPBackend.MATH,
    #                 [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH],
    #                 [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    #                 [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    #             ]
    #             for batch_size in [1, 5]
    #         ]
    #         fail_cases = []

    #         for use_mask, output_attentions, sdpa_backend, batch_size in cases:
    #             processed_inputs = inputs_dict.copy()

    #             # convert to ms_dtype
    #             if "pixel_values" in processed_inputs:
    #                 processed_inputs["pixel_values"] = processed_inputs["pixel_values"].to(dtype=ms_dtype)

    #             # slice for different batch sizes
    #             for key in ["pixel_values", "input_ids", "attention_mask"]:
    #                 if key in processed_inputs:
    #                     processed_inputs[key] = processed_inputs[key][:batch_size]

    #             # set attention mask with left padding
    #             if not use_mask:
    #                 processed_inputs.pop("attention_mask", None)
    #             else:
    #                 dummy_attention_mask = processed_inputs["attention_mask"]
    #                 dummy_attention_mask[:] = 1
    #                 dummy_attention_mask[:, :1] = 0
    #                 processed_inputs["attention_mask"] = dummy_attention_mask

    #             processed_inputs["output_attentions"] = output_attentions
    #             processed_inputs["output_hidden_states"] = True

    #             current_case = (
    #                 f"padding_side=left, use_mask={use_mask}, batch_size={batch_size}, sdpa_backend={sdpa_backend}"
    #             )

    #             prepared_inputs = self._prepare_for_class(processed_inputs, model_class)

    #             try:
    #                 with sdpa_kernel(sdpa_backend):
    #                     outputs_eager = model_eager(**prepared_inputs)
    #                     outputs_sdpa = model_sdpa(**prepared_inputs)
    #             except Exception as e:
    #                 fail_cases.append(f"{current_case}: {e}")
    #                 continue

    #             for key in logit_keys:
    #                 eager_logits = outputs_eager[key]
    #                 sdpa_logits = outputs_sdpa[key]

    #                 if use_mask:
    #                     eager_logits = eager_logits[:, 1:]
    #                     sdpa_logits = sdpa_logits[:, 1:]

    #                 is_close = np.allclose(eager_logits.asnumpy(), sdpa_logits.asnumpy(), atol=atol, rtol=rtol)
    #                 if not is_close:
    #                     fail_cases.append(get_mean_reldiff(key, current_case, sdpa_logits, eager_logits, atol, rtol))

    #         self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))


class SiglipVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in ViT, the seq length equals the number of patches
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches

    # Copied from tests.models.clip.test_modeling_clip.CLIPVisionModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return SiglipVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = SiglipVisionModel(config=config)
        model.set_train(False)
        result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    # Copied from tests.models.clip.test_modeling_clip.CLIPVisionModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_mindspore
class SiglipVisionModelTest(SiglipModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SIGLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SiglipVisionModel,) if is_mindspore_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["SiglipMultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False

    def setUp(self):
        self.model_tester = SiglipVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=SiglipVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SIGLIP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SiglipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="SiglipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/siglip-base-patch16-224"
        model = SiglipVisionModel.from_pretrained(model_name, from_pt=True)
        self.assertIsNotNone(model)

    @parameterized.expand([("float16",), ("float32",)])
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, ms_dtype: str):
        super().test_eager_matches_sdpa_inference(
            ms_dtype=ms_dtype,
            logit_keys=("pooler_output", "last_hidden_state"),
            use_attention_mask_options=(False,),
        )

    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class SiglipTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :int(start_index)] = 1
                input_mask[batch_idx, int(start_index):] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return SiglipTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = SiglipTextModel(config=config)
        model.set_train(False)
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_mindspore
class SiglipTextModelTest(SiglipModelTesterMixin, unittest.TestCase):
    all_model_classes = (SiglipTextModel,) if is_mindspore_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    model_split_percents = [0.5, 0.8, 0.9]

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.setUp with CLIP->Siglip
    def setUp(self):
        self.model_tester = SiglipTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SiglipTextConfig, hidden_size=37)

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_config
    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="SiglipTextModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="SiglipTextModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipTextModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipTextModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip does not use inputs_embeds")
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SiglipTextModel has no base class and is not available in MODEL_MAPPING")
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_save_load_fast_init_from_base
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="SiglipTextModel has no base class and is not available in MODEL_MAPPING")
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_save_load_fast_init_to_base
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/siglip-base-patch16-224"
        model = SiglipTextModel.from_pretrained(model_name, from_pt=True)
        self.assertIsNotNone(model)

    @parameterized.expand([("float16",), ("float32",)])
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, ms_dtype: str):
        super().test_eager_matches_sdpa_inference(
            ms_dtype=ms_dtype,
            logit_keys=("pooler_output", "last_hidden_state"),
            use_attention_mask_options=(False, True),
        )

    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class SiglipModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = SiglipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = SiglipVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return SiglipConfig.from_text_vision_configs(
            self.text_model_tester.get_config(),
            self.vision_model_tester.get_config(),
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = SiglipModel(config).set_train(False)
        result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": False,
        }
        return config, inputs_dict


@require_mindspore
class SiglipModelTest(SiglipModelTesterMixin, unittest.TestCase):
    all_model_classes = (SiglipModel,) if is_mindspore_available() else ()
    pipeline_model_mapping = {"feature-extraction": SiglipModel} if is_mindspore_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["SiglipMultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = SiglipModelTester(self)

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_hidden_states_output
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_retain_grad_hidden_states_attentions
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="SiglipModel does not have input/output embeddings")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_model_get_set_embeddings
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_load_vision_text_config with CLIP->Siglip
    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save SiglipConfig and check if we can load SiglipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = SiglipVisionConfig.from_pretrained(tmp_dir_name, force_download=True)
            self.assertDictEqual(
                config.vision_config.to_dict(), 
                vision_config.to_dict(), 
                msg=f"""SigLIPConfig.to_dict():{config.to_dict()},
                SigLIPConfig.vision_config.to_dict():{config.vision_config.to_dict()},
                PretrainedConfig.to_dict():{vision_config.to_dict()},
                tmp_dir_name: {tmp_dir_name},
                """
            )

        # Save SiglipConfig and check if we can load SiglipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = SiglipTextConfig.from_pretrained(tmp_dir_name, force_download=True)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/siglip-base-patch16-224"
        model = SiglipModel.from_pretrained(model_name, from_pt=True)
        self.assertIsNotNone(model)
        
    @parameterized.expand([("float16",), ("float32",)])
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, ms_dtype: str):
        super().test_eager_matches_sdpa_inference(
            ms_dtype=ms_dtype,
            logit_keys=("logits_per_image", "logits_per_text", "image_embeds", "text_embeds"),
            use_attention_mask_options=(False, True),
        )

    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


class SiglipForImageClassificationModelTester(SiglipModelTester):
    def __init__(self, parent):
        super().__init__(parent)
        self.batch_size = self.vision_model_tester.batch_size
        self.num_hidden_layers = self.vision_model_tester.num_hidden_layers
        self.hidden_size = self.vision_model_tester.hidden_size
        self.seq_length = self.vision_model_tester.seq_length

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_mindspore
class SiglipForImageClassificationModelTest(SiglipModelTesterMixin, unittest.TestCase):
    all_model_classes = (SiglipForImageClassification,) if is_mindspore_available() else ()
    pipeline_model_mapping = {"image-classification": SiglipForImageClassification} if is_mindspore_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    # MP works but offload doesn't work when the MultiheadAttention is offloaded
    # TODO: One potential solution would be to add to set preload_module_classes = ["SiglipMultiheadAttentionPoolingHead"]
    # in the dispatch_model function
    test_cpu_offload = False
    test_disk_offload_safetensors = False
    test_disk_offload_bin = False
    _is_composite = True

    def setUp(self):
        self.model_tester = SiglipForImageClassificationModelTester(self)

    @unittest.skip(reason="SiglipForImageClassification does not support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support inputs_embeds")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass
    
    @parameterized.expand([("float16",), ("float32",)])
    @slow
    @is_flaky()
    def test_eager_matches_sdpa_inference(self, ms_dtype: str):
        super().test_eager_matches_sdpa_inference(
            ms_dtype=ms_dtype, logit_keys=("logits",), use_attention_mask_options=(False,)
        )

    def test_sdpa_can_dispatch_composite_models(self):
        super().test_sdpa_can_dispatch_composite_models()


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_mindspore
class SiglipModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "google/siglip-base-patch16-224"
        model = SiglipModel.from_pretrained(model_name, from_pt=True)
        processor = SiglipProcessor.from_pretrained(model_name, from_pt=True)

        image = prepare_img()
        inputs = processor(
            text=["a photo of 2 cats", "a photo of 2 dogs"], images=image, padding="max_length", return_tensors="ms"
        )

        # forward pass
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # verify the logits
        self.assertEqual(
            logits_per_image.shape,
            (inputs.pixel_values.shape[0], inputs.input_ids.shape[0]),
        )
        self.assertEqual(
            logits_per_text.shape,
            (inputs.input_ids.shape[0], inputs.pixel_values.shape[0]),
        )

        expected_logits = mindspore.tensor([[-0.7567, -10.3354]])
        
        self.assertTrue(np.allclose(outputs.logits_per_image.asnumpy(), expected_logits.asnumpy(), atol=1e-1))

        # verify the probs
        probs = ops.sigmoid(logits_per_image)  # these are the probabilities
        expected_probs = mindspore.tensor([[3.1937e-01, 3.2463e-05]])
        self.assertTrue(np.allclose(probs.asnumpy(), expected_probs.asnumpy(), atol=1e-1))

    @slow
    def test_inference_interpolate_pos_encoding(self):
        model_name = "google/siglip-base-patch16-224"
        model = SiglipModel.from_pretrained(model_name, from_pt=True)

        # 640 x 480 image
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        processor = SiglipProcessor.from_pretrained(model_name, do_resize=False, size={"height": 480, "width": 640}, from_pt=True)

        inputs = processor(text="what's in the image", images=image, return_tensors="ms")

        # forward pass
        outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the shape
        # patch size = 16
        # batch size 1, (640/16) * (480/16) = 1200 patches, 768 hidden size
        expected_shape = (1, 1200, 768)

        self.assertEqual(outputs.vision_model_output.last_hidden_state.shape, expected_shape)
