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
# ============================================
"""Testing suite for the Mindspore VisionTextDualEncoder model."""

import collections
import tempfile
import unittest

import numpy as np

from mindnlp.utils.testing_utils import (require_mindspore, require_vision, slow,
                                         is_vision_available, is_mindspore_available)


from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from ..bert.test_modeling_bert import BertModelTester
from ..clip.test_modeling_clip import CLIPVisionModelTester
from ..deit.test_modeling_deit import DeiTModelTester
from ..roberta.test_modeling_roberta import RobertaModelTester
from ..vit.test_modeling_vit import ViTModelTester


if is_mindspore_available():
    import mindspore as ms

    from mindnlp.transformers import (
        BertModel,
        CLIPVisionModel,
        DeiTModel,
        RobertaModel,
        VisionTextDualEncoderConfig,
        VisionTextDualEncoderModel,
        ViTModel,
    )

if is_vision_available():
    from PIL import Image

    from mindnlp.transformers import VisionTextDualEncoderProcessor


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


@require_mindspore
class VisionTextDualEncoderMixin:
    def get_vision_text_model(self, config, text_config):
        pass

    def prepare_config_and_inputs(self):
        pass

    def get_pretrained_model_and_inputs(self):
        pass

    def check_model_from_pretrained_configs(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)

        model = VisionTextDualEncoderModel(config)
        model.set_train(False)

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], config.projection_dim))

    def check_vision_text_dual_encoder_model(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.set_train(False)

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_vision_text_dual_encoder_from_pretrained(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        kwargs = {"vision_model": vision_model, "text_model": text_model}
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(**kwargs)
        model.set_train(False)

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

        self.assertEqual(output["text_embeds"].shape, (input_ids.shape[0], model.config.projection_dim))
        self.assertEqual(output["image_embeds"].shape, (pixel_values.shape[0], model.config.projection_dim))

    def check_save_load(self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.set_train(False)

        output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        out_1 = output[0].numpy()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            model = VisionTextDualEncoderModel.from_pretrained(tmpdirname).set_train(False)

            after_output = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            out_2 = after_output[0].numpy()
            max_diff = np.amax(np.abs(out_2 - out_1))
            self.assertLessEqual(max_diff, 1e-5)

    def check_vision_text_output_attention(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.set_train(False)

        output = model(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=True
        )

        vision_attentions = output.vision_model_output.attentions
        self.assertEqual(len(vision_attentions), vision_config.num_hidden_layers)

        # in ViT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(vision_model.config.image_size)
        patch_size = to_2tuple(vision_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        self.assertEqual(vision_attentions[0].shape[-3:], (vision_config.num_attention_heads, seq_len, seq_len))

        text_attentions = output.text_model_output.attentions
        self.assertEqual(len(text_attentions), text_config.num_hidden_layers)

        self.assertEqual(
            text_attentions[0].shape[-3:],
            (text_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]),
        )

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    def test_vision_text_dual_encoder_model(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_dual_encoder_model(**inputs_dict)

    def test_model_from_pretrained_configs(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_model_from_pretrained_configs(**inputs_dict)

    def test_vision_text_dual_encoder_from_pretrained(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_dual_encoder_from_pretrained(**inputs_dict)

    def test_save_load(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_save_load(**inputs_dict)

    def test_vision_text_output_attention(self):
        inputs_dict = self.prepare_config_and_inputs()
        self.check_vision_text_output_attention(**inputs_dict)

    @slow
    def test_real_model_save_load_from_pretrained(self):
        model_2, inputs = self.get_pretrained_model_and_inputs()

        outputs = model_2(**inputs)
        out_2 = outputs[0].numpy()

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model_2.save_pretrained(tmp_dirname)
            model_1 = VisionTextDualEncoderModel.from_pretrained(tmp_dirname)

            after_outputs = model_1(**inputs)
            out_1 = after_outputs[0].numpy()
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)


@require_mindspore
class ViTBertModelTest(VisionTextDualEncoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            vision_model_name_or_path="hf-internal-testing/tiny-random-vit",
            text_model_name_or_path="hf-internal-testing/tiny-bert"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.vision_model.config.num_channels,
                model.vision_model.config.image_size,
                model.vision_model.config.image_size,
            ]
        )
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return model, inputs

    def get_vision_text_model(self, vision_config, text_config):
        vision_model = ViTModel(vision_config).set_train(False)
        text_model = BertModel(text_config).set_train(False)
        return vision_model, text_model

    def prepare_config_and_inputs(self):
        vit_model_tester = ViTModelTester(self)
        bert_model_tester = BertModelTester(self)
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values, _ = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }


@require_mindspore
class DeiTRobertaModelTest(VisionTextDualEncoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            vision_model_name_or_path="hf-internal-testing/tiny-random-deit",
            text_model_name_or_path="hf-internal-testing/tiny-random-roberta"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.vision_model.config.num_channels,
                model.vision_model.config.image_size,
                model.vision_model.config.image_size,
            ]
        )
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return model, inputs

    def check_vision_text_output_attention(
        self, text_config, input_ids, attention_mask, vision_config, pixel_values=None, **kwargs
    ):
        vision_model, text_model = self.get_vision_text_model(vision_config, text_config)
        model = VisionTextDualEncoderModel(vision_model=vision_model, text_model=text_model)
        model.set_train(False)

        output = model(
            input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, output_attentions=True
        )

        vision_attentions = output.vision_model_output.attentions
        self.assertEqual(len(vision_attentions), vision_config.num_hidden_layers)

        # in DEiT, the seq_len equals the number of patches + 2 (we add 2 for the [CLS] and distillation tokens)
        image_size = to_2tuple(vision_model.config.image_size)
        patch_size = to_2tuple(vision_model.config.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 2
        self.assertEqual(vision_attentions[0].shape[-3:], (vision_config.num_attention_heads, seq_len, seq_len))

        text_attentions = output.text_model_output.attentions
        self.assertEqual(len(text_attentions), text_config.num_hidden_layers)

        self.assertEqual(
            text_attentions[0].shape[-3:],
            (text_config.num_attention_heads, input_ids.shape[-1], input_ids.shape[-1]),
        )

    def get_vision_text_model(self, vision_config, text_config):
        vision_model = DeiTModel(vision_config).set_train(False)
        text_model = RobertaModel(text_config).set_train(False)
        return vision_model, text_model

    def prepare_config_and_inputs(self):
        vit_model_tester = DeiTModelTester(self)
        bert_model_tester = RobertaModelTester(self)
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values, _ = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }

    # skip as DeiT is not available in Flax
    def test_pt_flax_equivalence(self):
        pass


@require_mindspore
class CLIPVisionBertModelTest(VisionTextDualEncoderMixin, unittest.TestCase):
    def get_pretrained_model_and_inputs(self):
        model = VisionTextDualEncoderModel.from_vision_text_pretrained(
            vision_model_name_or_path="hf-internal-testing/tiny-random-clip",
            text_model_name_or_path="hf-internal-testing/tiny-bert"
        )
        batch_size = 13
        pixel_values = floats_tensor(
            [
                batch_size,
                model.vision_model.config.num_channels,
                model.vision_model.config.image_size,
                model.vision_model.config.image_size,
            ]
        )
        input_ids = ids_tensor([batch_size, 4], model.text_model.config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return model, inputs

    def get_vision_text_model(self, vision_config, text_config):
        vision_model = CLIPVisionModel(vision_config).set_train(False)
        text_model = BertModel(text_config).set_train(False)
        return vision_model, text_model

    def prepare_config_and_inputs(self):
        clip_model_tester = CLIPVisionModelTester(self)
        bert_model_tester = BertModelTester(self)
        vision_config_and_inputs = clip_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }


@require_vision
@require_mindspore
class VisionTextDualEncoderIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model = VisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian", logit_scale_init_value=1.0, ignore_mismatched_sizes=True)
        processor = VisionTextDualEncoderProcessor.from_pretrained("clip-italian/clip-italian", ignore_mismatched_sizes=True)

        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        inputs = processor(
            text=["una foto di un gatto", "una foto di un cane"], images=image, padding=True, return_tensors="ms"
        )

        outputs = model(**inputs)

        # verify the logits
        self.assertEqual(outputs.logits_per_image.shape, (inputs.pixel_values.shape[0], inputs.input_ids.shape[0]))
        self.assertEqual(
            outputs.logits_per_text.shape,
            (inputs.input_ids.shape[0], inputs.pixel_values.shape[0]),
        )

        expected_logits = ms.tensor([[1.2284727, 0.3104122]])

        self.assertTrue(np.allclose(outputs.logits_per_image.asnumpy(), expected_logits.asnumpy(), atol=1e-3))