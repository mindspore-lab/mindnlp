# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the mindspore ViLT model."""
import io
# pylint: disable=W0231
# pylint: disable=E1102

import unittest

from datasets import load_dataset
from packaging import version

from mindspore import ops
from mindnlp.transformers import ViltConfig, ViltProcessor
from mindnlp.utils.testing_utils import (
    require_mindspore,
    require_vision,
    is_vision_available,
    is_mindspore_available,
)
from mindnlp.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask

if is_mindspore_available():
    import mindspore

    from mindnlp.transformers.models.vilt import (
        ViltForImageAndTextRetrieval,
        ViltForImagesAndTextClassification,
        ViltForMaskedLM,
        ViltForQuestionAnswering,
        ViltForTokenClassification,
        ViltModel,
    )
    from mindnlp.transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

if is_vision_available():
    import numpy as np
    import PIL
    from PIL import Image




class ViltModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        image_size=30,
        patch_size=2,
        num_channels=3,
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
        modality_type_vocab_size=2,
        add_multiple_images=False,
        num_images=-1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
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
        self.modality_type_vocab_size = modality_type_vocab_size
        self.add_multiple_images = add_multiple_images
        self.num_images = num_images
        # we set the expected sequence length (which is used in several tests)
        # this is equal to the seq length of the text tokens + number of image patches + 1 for the CLS token
        self.expected_seq_len = self.seq_length + (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        if self.add_multiple_images:
            pixel_values = floats_tensor([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        else:
            pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return (config, input_ids, token_type_ids, input_mask, pixel_values, token_labels)

    def get_config(self):
        return ViltConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
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
            num_labels=self.num_labels,
            modality_type_vocab_size=self.modality_type_vocab_size,
            num_images=self.num_images,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        pixel_values,
        token_labels,
    ):
        model = ViltModel(config=config)
        model.set_train(False)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, pixel_values=pixel_values)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.expected_seq_len, self.hidden_size)
        )

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        pixel_values,
        token_labels,
    ):
        model = ViltForTokenClassification(config=config)
        model.set_train(False)
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, token_type_ids=token_type_ids, pixel_values=pixel_values)
        result = model(input_ids, pixel_values=pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            pixel_values,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
        }
        return config, inputs_dict

    def prepare_pixel_values(self):
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])


@require_mindspore
class ViltModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            ViltModel,
            ViltForQuestionAnswering,
            ViltForImageAndTextRetrieval,
            ViltForMaskedLM,
            ViltForTokenClassification,
        )
        if is_mindspore_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-feature-extraction": ViltModel, "visual-question-answering": ViltForQuestionAnswering}
        if is_mindspore_available()
        else {}
    )
    test_pruning = False
    test_headmasking = False
    test_mindsporescript = False
    model_split_percents = [0.5, 0.8, 0.9]

    # ViltForMaskedLM, ViltForQuestionAnswering and ViltForImagesAndTextClassification require special treatment
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class.__name__ == "ViltForQuestionAnswering":
                inputs_dict["labels"] = ops.zeros(
                    self.model_tester.batch_size, self.model_tester.num_labels
                )
            elif model_class.__name__ in ["ViltForMaskedLM", "ViltForTokenClassification"]:
                inputs_dict["labels"] = ops.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=mindspore.int64
                )
            elif model_class.__name__ == "ViltForImagesAndTextClassification":
                inputs_dict["labels"] = ops.zeros(
                    self.model_tester.batch_size, dtype=mindspore.int64
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = ViltModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViltConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                config.modality_type_vocab_size = 3

            # ViltForImageAndTextRetrieval doesn't support training for now
            if model_class.__name__ in [*MODEL_MAPPING_NAMES.values(), "ViltForImageAndTextRetrieval"]:
                continue

            model = model_class(config)
            model.set_train(True)
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            for k, v in inputs.items():
                print(k, v.shape)
            loss = model(**inputs).loss

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_save_load(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_determinism(self):
        pass

    @unittest.skip(
        "VilT samples image tokens from a multinomial distribution, resulting in not deterministic hidden states"
    )
    def test_batching_equivalence(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states"""
    )
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(
        reason="""VilT samples image tokens from a multinomial distribution, resulting in not deterministic
                            hidden states. Cannot test equivalence on logit level"""
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "expected_seq_len", None)

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.set_train(False)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            if model_class.__name__ == "ViltForImagesAndTextClassification":
                # attentions are a list of length num_images
                # each element contains the attentions of a particular image index
                self.assertEqual(len(attentions), self.model_tester.num_images)
                self.assertEqual(len(attentions[0]), self.model_tester.num_hidden_layers)
            else:
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.set_train(False)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            if model_class.__name__ == "ViltForImagesAndTextClassification":
                # attentions are a list of length num_images
                # each element contains the attentions of a particular image index
                self.assertEqual(len(attentions), self.model_tester.num_images)
                self.assertEqual(len(attentions[0]), self.model_tester.num_hidden_layers)
            else:
                self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                self.assertListEqual(
                    list(attentions[0][0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.set_train(False)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                self.assertEqual(len(self_attentions), self.model_tester.num_images)
                self.assertEqual(len(self_attentions[0]), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0][0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )
            else:
                self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, seq_len, seq_len],
                )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.set_train(False)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            if model_class.__name__ == "ViltForImagesAndTextClassification":
                # hidden_states are a list of length num_images
                # each element contains the hidden states of a particular image index
                self.assertEqual(len(hidden_states), self.model_tester.num_images)
                self.assertEqual(len(hidden_states[0]), expected_num_layers)
            else:
                self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.expected_seq_len

            if model_class.__name__ == "ViltForImagesAndTextClassification":
                self.assertListEqual(
                    list(hidden_states[0][0].shape[-2:]),
                    [seq_length, self.model_tester.hidden_size],
                )
            else:
                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [seq_length, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            print("Model class:", model_class)
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip("MindSpore has no .grad")
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = True

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        attentions = outputs.attentions[0]

        if model_class.__name__ == "ViltForImagesAndTextClassification":
            # hidden_states are a list of length num_images
            # each element contains the hidden states of a particular image index
            hidden_states[0].retain_grad()
            attentions[0].retain_grad()
        else:
            hidden_states.retain_grad()
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        if model_class.__name__ == "ViltForImagesAndTextClassification":
            # hidden_states are a list of length num_images
            # each element contains the hidden states of a particular image index
            self.assertIsNotNone(hidden_states[0].grad)
            self.assertIsNotNone(attentions[0].grad)
        else:
            self.assertIsNotNone(hidden_states.grad)
            self.assertIsNotNone(attentions.grad)

    #@slow
    def test_model_from_pretrained(self):
        model_name = "dandelin/vilt-b32-mlm"
        model = ViltModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_mindspore
class ViltForImagesAndTextClassificationModelTest(ViltModelTest, unittest.TestCase):
    all_model_classes = (ViltForImagesAndTextClassification,) if is_mindspore_available() else ()

    def setUp(self):
        self.model_tester = ViltModelTester(self, modality_type_vocab_size=3, add_multiple_images=True, num_images=2)
        self.config_tester = ConfigTester(self, config_class=ViltConfig, hidden_size=37)

    @unittest.skip("We only test the model that takes in multiple images")
    def test_model(self):
        pass

    @unittest.skip("We only test the model that takes in multiple images")
    def test_for_token_classification(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")

    return image

def hex_to_image(hex_str):
    image_bytes = bytes.fromhex(hex_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

@require_mindspore
@require_vision
class ViltModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa") if is_vision_available() else None


    def test_inference_masked_lm(self):
        model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of [MASK] laying on a [MASK]."
        inputs = processor(image, text, return_tensors="ms")

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape = mindspore.ops.shape(mindspore.Tensor(np.ones(shape=[1, 11, 30522]), mindspore.float32))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = mindspore.tensor([-12.5061, -12.5123, -12.5174])
        logits_slice_np = outputs.logits.asnumpy()[0, 0, :3]
        expected_slice_np = expected_slice.asnumpy()
        self.assertTrue(np.allclose(logits_slice_np, expected_slice_np, atol=0.5))

        # verify masked token prediction equals "cats"
        predicted_id = outputs.logits[0, 4, :].argmax(-1).item()
        assert processor.decode([predicted_id]) == "cats"


    def test_inference_visual_question_answering(self):
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        processor = self.default_processor
        image = prepare_img()
        text = "How many cats are there?"
        inputs = processor(image, text, return_tensors="ms")

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape = mindspore.ops.shape(mindspore.Tensor(np.ones(shape=[1, 3129]), mindspore.float32))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = mindspore.tensor([-15.9495, -18.1472, -10.3041])
        logits_np = outputs.logits[0, :3].asnumpy()
        expected_slice_np = expected_slice.asnumpy()

        self.assertTrue(np.allclose(logits_np, expected_slice_np, atol=0.3))

        # compute loss
        vqa_labels = [[2, 3, 155, 800]]
        vqa_scores = [[1.0, 0.3, 0.3, 0.3]]
        labels = ops.zeros(1, model.config.num_labels)

        for i, (labels_example, scores_example) in enumerate(zip(vqa_labels, vqa_scores)):
            for l, s in zip(labels_example, scores_example):
                labels[i, l] = s

        # forward pass
        outputs = model(**inputs, labels=labels)

        # verify we have a positive loss
        self.assertTrue(outputs.loss > 0)

    def test_inference_natural_language_visual_reasoning(self):
        model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

        processor = self.default_processor

        image1_hex = "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d0d1832211c213232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232ffc000110800d9014003012200021101031101ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f38f116f7bf0cea5722b1e4501335d6f8ae25660fe64591cfdfc9fc8573d1d83caa08f999ba469f331ff000fc688a1328db456d3cbe5dc0719c6d6438c7a8ad15d123520c704b723fd89b07ea01e0fe75a161a57d9a441e48924c82c7770bd46013c1c7527a56c59cce2521655500e4157cb21c639c73cf27d8d3482da1cd4ba51b81147112a5502b2cd84923519f9981eaa3764e3a566ddd83da5f189b7978488d97cb6c2903d71effc8d7a14aa5e58d564630a0dcdb9f3b1f3f7b27a13f2f5f71ce4d51bfd22cee637dd74b095fe2f30a05c9e393c1ff229a891b23828edde20aa55b68c31f9796527b7e1572359ac16311ae5e5244478c63b13fe15bd79a25ca08e711be11768909e0e063a8e3159b77a7c86de128f24ab1aed528a48e792723df152e3d09ea569879654cc3742e47eedfe6604f518ec73d39a8a48e0984f1a895235188c807e63effe7bd6ad959a6c995e55903107648bf31e319e48393d3af6148fa7b456ad81e66d24e33c839033c9e383ef9a12ea5a49941f42bf8b4e2d3db48234fdec4fc152a7ef0c827d8f34f8659923494af9b0c8a16503d47461ef8ad0b0636538f2a27e4879121271263939f5f5a9a4b19e0b8bc82e4efcb72d819f552401df383f5a6292bb326e9a58ce03065232afb07cc3fcf6a82cd244ba59e45ce55b6818c9e83fad5ff29043272d200d851bb72a9e9f507daa51a7969c80db805077c6d951d323d41c7b50c695b46673ba994e491f41d2a9cf312a5413b7fcf35b7269b3c8e44223922dd850fc37e7fe4552bad22ed41d968dd8801b3f867bd2b0d23a0f035c35bda5f3a22b3b488bb9870005cff5aeb05f48d1933c437631f2923f235e7ba6c9a969f13451c3b3cd70e43311918c74008fcf9e2ba0596490995991954fceac3072067271d7fc2981b5fda51e18979964246170491ce3d39ef5616f5268cb891d067037ae339f6ac1859e275d96ec14e5808e46ea7a1e460f3f9569417767290859a39d87dd923c1e9f91fc281976494296668d700005874fca9c8b24cab809838f9b3c9fc09a80089c2e11776ec74c11dbf038fe7446912a108ea08eeef9e3f1ff00f55022494794edbd652d9ced0a3a7d69d2498c1132ab039da5464fb1ff001a82620e51a5463b70704027f2355c1b65b431cb70bb02805cb0cfe278a432fc5704c7bb70e0f1b0600f7e6a52d0b164381ea4707f1e3354952de2508ad21e73eb834824f2dbf788c14679c8231f9d005fdb1aab167da80673bb39a41aa58c4540949cb754039fcea8a5cfee1f6a1403ae00c9f43c9cf6a9d350919d545bb3f7f9581fccf6a00d0b5b81227eee4756cf248207e7e952069b61501c32824f0318aa66668d43a29538e372e738aa13ea37c1331e59c9e810a023e99e690cd254bb9b0e618c751965527d3a7a75ab8ae91a6d752ce06d271c67a6462b969e3b8b99137f001c8f90b6dfd6a5567255d65918463a2a9033f4f4ff000a5703a00a8d17cb0b063f7baf1fafeb521b32d102ae6323a2a11c638fc6b105cdc82ce372b8393bd7248c74e957a3bb77b71209d8b0006c8d4923f3edd3f2a605e49e268d4ace8c41c64bf423822a418605c829212064107009e9598932a2b3ce7cb763805f058fa120723da98d6b7db58c3217623aae5b007afbff005a4d858d861146374caac00fba467db807e9546796c6572c675322f202b851803807f0aa10d95d5c1126f91508182eb9ce3d734f8f4fbdde42449bc13cb4602e7d7f1a4d8ec69472c289ba49adfcd71905496ce3a77e6a16b98862496f249581076850a3a74c8e71ef55ad2dcdc4537902299c06b694809c107e6009e9d3a8c7e7447a5ea4af94821893d4ca093f9d0ee091c5f8812e4be1806c7f73a5334af10c369a7fd967f93cb523788c3151fde1cf5f5245497ba45edccd142f1c48f29382dca280092cce381d3a77aa6346bad227f3034535cb8c4076fcb1923ef15e7247607033cf3815ad8846d6a1ad59cf32c7652acb6ea76be23e391c95c8e9927f1e6a8398751f3204b568e488e481c3819cf0475e3d0d58b7d091edadd03c64851f38c9f9b1cf39fc7e6a9574ff00b31092a4a15866de47cab1e7008e320819e79eb4b61bd434eb6b9824542c268cb613cc6cb203fc2c4f3b4f1d78ce0d4d2decf009ccb03fd9e39196438fde4641e8cbdf031c8c1c608dd9ad7b3816c6ccf9b34b7064623f7d20c292391d33ee7f2a82ed5ad3cc7f365557555478f05d39e793f7940ec41c638eb557b9362bdbc8ac1658a078411b92e2d64c249c746c639fa8fc2a196d2e56e3cc4b5b7b8908c960444e403fde51fa15fc6ad5b5dee6919364729c8660815580fbacc87d40e0f51d3b54b1996fe1f35650507dd298224e9db9c0fa1143681231fec6e6ee2b8bab3996119cee019c01ec8727ea39f6a77d9e09b732ba488e483b4640627807bf1ebdab51cb92184726d4393f39208f6fce926b1b394c9726dc24c0619e50010dedce78cff3a9f30b19896368d11fb4bb4170a1fcab52dfbc639c0618206dee48cfa7b54f6f85b1786598f99c2453aa798a130321b0723d88239c71eb6504accb0902e60118dcaf9619f5079ebfe7351dbc56ff0069921895442c0294df82141e7fc284c654b6b15b691e56b69e50904899405429607e6071c9ff0013d31546d634b7985d3183ed649409687cc38233f37f08047624e4115b5fd856770c0cb034ac0ee532283d0e723bd598e116b1b2c388d58e406018673ee3bd0162b5bc27ec81bca0157901d36b283dbeb9f5ed53b5b07448c4076b03b8a818c7f8d5a696251e5cf3859319242e47d38a9cdaa21e26de4f21782b83edf8f6a4332a4d363680ef842a3139608723dc8152268cd0230b6444563b89ddbb27f1ebff00d7ad1589e17dde6b4880e446c09fff00513fd29ef0a18c940b00277657839fa77a680cf78d94ed36ccdb46405201c8f73c506dc6d0ca8ca08c91c617dfa8ad140140728c5b6f1b4923dff1a192dcb8dfb189e7e724e7e83ffd5480a11401b82c863201c2b0249efd39a55b40cdb5164cfdecfb7bd6849690a02f82abb73c71f5fa76a8e2b585ce7cd39208da6427ff00d740110b508806e848ce3e7a8deced96440c58993e5c2444a83efc607e35721b677261b7b7dcdb81fddae73ee6936980c8d72a55b70f94b96dbdb81dbf2a2e162bc91a9668d0677721989e4d2c5a7aac4549651d8703a54cec230db97677e5b6e6a450eca58832018ec00fc280231047f2ed8d086c9076f7fae3de86499647daaaca064ee6c0fcbb9a77da2e1b0ab6d215e7716e98fc7f954caef2a64905f230bc600f6c7f5a00a0e279970c98e721c11daa292d5a45da8e15bab1ddfe715afe4968983381f374c13f8d2b44fcee70587191f2e47d290ccfb4b0207259e550157208fa7a55bfb39540d2336719c05ce3fc6aec71b35b985dc0c775e323f5f7a892cc44c5bed00b6738278c63a7f2f5a0089551f21937e01c9ddfcbbe2a4682168362c2c9c924a9e7f5ab1e5b49b910ae4f7039cd55904d0021942e78c062734c411e9cbe7a346a0a2803246081dfa77ef8ab3e4ac6c420394e770e48cf4c567cd717222f95d23908e84e4afae483d71525bdccb0ab994bbb9c046e791ea334b4196a3599914191ddba10e3ef7af03a719abf0858902b19194f424e00fc2b39659ee61658e6d84b6013ce47d3b5245a75da6d33df3be49254a336589e99cfe940176282d6c2dcc3631c299cb151c02c7fc4d57f36468f75dc36e133c2921b9cf73435bf9192817cb41f3313c2f7ee6905c88e74599ffd1dd5b73e4eeddc60631df27272318efd8033de408046f03cea5b01526ce3f0efebcf1cd604d6805eb47f637495811b8260638cf7f5e3815bd6d69f6728c034a7a311c01c9f9c13ee6ab3a94bb6fb4dc6e73b82451282aa3b1f73f5ee6a84654902ce9045244a0a31728ce4292463a0eb93c9047f2ad08d8da732c8acd1a611371f9067a7a0cf5ab52d8c6e736ab132900972a48c11cedddd0f6fc3d293ec654a93e7303d7f79c11fd3fa500512a1a3fddc64c81b85640c64e9dfb8e71f8d68c3a14f7a2df7c5201212802a8c273dfebc1fc2b7b47f0e7da1e39638a48da33c89242c49eff4e3041ef9aee6deca0b28ba0e467ffaf59ce5ca8a8abb387d37c180466497f74c81e342bd812318ff000abc9e1ed3ec8bb2461588dbf27cbbbd7a56beadabc76e92aa91b94eee2b8d9fc48eefbf3b62fe27ebf97f8d79f3ab26ceb841245fbab5d350fef2253cf181c935892e8f6afe6cf69710c521520895c053f561c8e83f2e958fac788c4c8c9117553fc4a393f5cd57d122b9b88e47de38fef2eefcc56b4272721548c546e5cbab29e505bed27112ed4119ebcf5e7f4f4eb4d992e638e68260d3c6ff007a477e7f2c6323d40fc2b5a1b3890a6d891491cec50179fe62a592448ce70c1570a367009c9c671fcfdebbf438cc682d1becc24b789e472010252db54679e38c9c7f4a98c6b296695cb321cb80a3683e83d7d38ab97124cdb8e014f539cfe359de4cfe6092390c6adf29468f83efcf7a00b90451b5b2b08cc64f40700afb53962264242a30032369048fcff9d317cd2bbd570493c95ce31e829a91ddca8cfe76d8c60e15546ecfb633fad202c973021320384e8483dfb5093c323aeef949e80f1fd2a230a6e3b943161cef3ffd7a66d67915fca2a38da0373dfaf07f0a00b72bbf96762a850b9207383f875a238923c91fbb90e4862723f2ed490ca59d9523e579240207ff005ba53b0e847ce6341804105b23da80228becc902867660df2866e41cff003353189e59fcb5381d4b15e02fafb9a16369583b2cb22f24031ed0a3b706ac98a68eda48d4049e4c0393caaff9c66b2ab55538dd9a52a6e72b19b79acc504df6550e2203927d7d4d22cff6f8d2e4c68619937ef66c1cfa62a8de689f7bcfb96503a9562a413dc9fe955d6f45a58da5bda2b4c17f769e637451dcf6cf7e6b869e292777b9d93c3b968b636e2313bb19a458608c6e665e5f078000ee49e00cf5a5fb618656db0e223caa48fb997db8ea73f87bd41a5cb697489179db8c6ef35c36ee4bfdd41f4519fcf35766b7b17e230642081c6481f9576d293947999c7522a2ec8a335cb99406279c64677671cfafbd4526a324110f914b742c54038ec715a906993c8a4e62191d110fcbf8e7d6a8df85d2648db5047684e71304f94b67b81ce307deadc92172b6518ef6e9e2561b162db8590c800cfd318eb4f845f0982fda14b2ae46d70723d319e7a7a7e356e6b27d4a25fb3959ed255eb16d2bf43dfd6ab43e1d58d5926458f763091b8fd053b8ac4ce6f99f0165c63a30fbdf87a0ab8b2dcf95fbce091c0c727e951a69d05929483cf0ec368791cb127be727ad5d9ac6e9e2061744980fb814fe87d6811247e4846982b93b70770239fc7f9d4ca2111b339e54004ee23a9e3bfbd53b5b3bf042cd0b2b06c932138f700e6a596de1b5b72260d31c82487ce31d064f145c2c5d30c71c2ca881f1ddcee623d73deaa2da0925f2a4470ac72a4b63b738f7a856ede69b31595d0666c1209207d38c7bff009357492aa0cf2caa33bd5719238f6a00516712b8313ba0073d896e3a0cf414f4b24707cc21941f53fce9d02c09082cecc491f3380a7f5f7aad7514934e4a4ae36748f25540e9d3bf434c0b11c1676f161522490f1fde3fad417167692469e7c46e02e06d5eddba0ed458d80da92ca0824fc9c6481ebf4ad0942c49b04819faaa95ce3bd2e8073d3436a970ac1490704fce58237b64e067daacc7142141c1539ce420fcf3486c9114b966fbbd5b2403eb8e3f2f7a7468c8725378c00020fbb803a0ce3be6ac431922970e8c5d50f05892a4fd2a5d3ed9ae6f1049b3e6eaa3a67db3de95504319dd81c81f28c927dcd69681145757c6392ddd82f1bb9071e878c1a019da69f6a2dad03e77103efe391f51593adea4b042e09c0ec476cf06b5eede3b5b4d8bbd4e30371cfe15e47e32d64422e10c857032141ebfe78ae3c4cddf94e8a115b98fad6be59551db71248f2d4e4b76009a65ae877574c27be944719191127503fa567784b4f3a8ddc97f3b12b0f0b9e72e79cfe03f9fb57632c8f0ccab24b1459036ee392e73fa7ff5a951a09abc8aab55c7dd4671d16d908916253b4f3bc927dbd6a76592db682aedb473bc850a3d7db1fad5c7784b6d6badcdc2b0438c63f977a8bccb764011f7ee27207cd91e9cff005aea8c547639dc9bdc6a406366984b9574c0dc00c1cf5ffeb53079865f28dc7c839648e3c71e9cd3dd9a0b20bb11958841938298f4c7d696de05b8469c9deb9cede0e3b707bd512243ba4de0471c51938cb9259bebe95718c2880b0550782aab9c5554924dfe5bf420e060e47e3d07ff005aa401bcc56208c70a38fc4e7af6a005f2d0b30329c8c649718f539ec38a63c0973b629629a48c1f95c36dddf91cff004a95777468c1561c6de473db15379802909df9cab673401562b68a18652f19544193925f208ec3927e9527d99210e11bef7206327dfa7ffaea444dcc371f2f664b166c2aaf724f41f5a5825572922f992237259405dbf81c1e68022b7b3b58f76e40649382dce493ce07a54acbe586011cb609f9cfa53d55a65205bdda10c47cdb00ef823e6c8f5a57867077b32ab93d4a86dbc743f8f349b495d8d26dd912db332b34af16e48f9018f53dbf5ac4d4a698c5717724cc1d0193048f9b9c77e833c0f5ab7a96a16b65a60b7b561279792cd8c658f2588f7ae5fedf24e9e6ccd1990ee73e6e1803c85e3b003a1fd2bcaad37565e47a7461ece268437925c69ef1a3b24aca09186ce71d7f3f7edd6b941a865ec6cda38c46a4b4922e19871d3eb5d9d9e9bf6a93c88ca19cc49e6ca14b02c4648273f4f4c7bd636b5a0c5a3f95706e598ae2350ca31b4fa71d88ee4d38c52073772ff83ef6192e26b59406762648dc8c920601fe86bb68ece79a35c2186dc3659ba1635e6fe0f8927f12db3c99b68256d8a4ff0013907033db3cfe95edb7924763a6aac64204f954b1fe2f5fc2ba613b42c72ca1efdce7efa2905a88ada205b3b40fee9f527f3e2a86afa3c8de1bbb86494120799f3118c8f4f7c1e95a73ddb41e4aee3f38dcc547af39a96e44726952995dd8be044a83976ebc7d2b3f335e963cd748d424d2ad92e951d2012347708e772128d838e3e538c11ea0d76f1436daac4b71613abe3a104654fa5735ace9ada25844a2252b73233488cdb838239c9cfdec0fd07a571f6baade784f599a2864dd0a1c6dcf041e9cfd0d0a525aa138c5e87af476b247283e5ee603249e3351224c652c4a797dcf9a49fc2aae9daf26a36cb3c64c9194c4919ebebfa7eb5a0f7533cabe54311b278d76cc18e59b273903ee8f7c9fa575d3a8a68e59c1c58a6dc3c8b2b3640ebdff00534cd913ee20a855e14e3240fa914f128224c1393d9802579edfe1529db7482012300a0367bff9f6ab206940b1fef24dea3a97cf1fc80aa7324e612ab6be78e32a0e4633dc679c7f9156e18197992e4bed27031dff003ab2af15ba976270064e17934018d0dbcc6e09fb2c1b8e573b3e6031d393d2a586ca617041541c672a08241f7fe9cd599352bae90c3146b9cee7c96dbc76f5e4f155a56b8b908ef732ed1db8e38c0c74f7a063652eb3baac730283972c76e0934a2de76603cb773c1c87e83af5ebeb4c6b391ccdb63795e4182a490a707afb77ad1b68e2b4b7113c9e64dd703240a046649fbbb82263e533fcaa589c8207180781c67f9d3e186592490c84a11cfc899dc07a7538fd695a5717a5d62491b2373fdd2c9dbae71d7b53e699449bdd01190546fe578ea703033f5a398394a4678e25ce4931ff007810a0fa60679ff1ab9a15dcb6d7a4bbb10dcec8c16cfaf5c67a8ff0aab7373650c6d2347e748dd0b0e58751cfb7ad675b4ffda575bbfb3a58486fbc0fdc1dd89cf03bfd3de8e6b8ec7a46b12f9963bba647a7f4af12f1b1604b141b7a67ff00ad5e8697934567e51cbc4df724e49cd705e304b868b12a058c9e1b039ae3c42f7b98e8a1b58ade02d42d5ec2eac1895bb8e469802d81246463f02a7afb11cd6dcfa2c52bf0d8ce092c7711dce4d795e66b1bb4b8b7729346db918763fe7b77e95dc5a7c4fd4678d14d85ac6d8f9d90b7cc7b9c76fa64d6f1a8b96e4ce9b72d0e9ed7499235db0db48622301c900b9fa9e715343a361e469aeae370e885f23d7d7e9d7ad6443e30bbbb5f96da38c9fe30cc7f4ab435fb994fef76a2e39d917f8e69baf044fb099ad3db471c5f777b81b7e6236ff9e6a316a801dd11dc7a6e3c01d80e7fc6a38eee595406590e0654e7dbd6a686e67e331ed6cf0c38cd672c5451a2c336442de42bc7cb160e0818c9f61d6a54b6625570d1ed5e8cc1b9f4e9d30695eee6191e68073d5bff00ad51adf5ccc842ca5c7e1c0fad4fd6e3d87f557dc7b44f1caacccabb810aa46724f07bfd2a58231b481ce064f2393ebc7d3a5521a94d061a5b6000e8db3273eb5760d61e4dc23446cf52a8071fce9fd6a3d05f569771d2c0d3210f03c884f059411fa8a9191c8d8a195581c918c7d4f7e688f5a7b69c09a491548c01d703d714977a8df4e04492f98ae32b842a4fb5258b56d83eacfb8a6e12de35591cf9bff3cd1b71f6e6b3ae2791be7dcc48f976e385fa565dd477d62c259202f183ca820303fd69f06bd1bda1450bbd9828f34952a7dab9aad77534d91d34a8c60afb9897cfe75c869048811be7238cff00877fc6b36d8cdacdec768bb332801b9c6549f5c8cf02a6d5adeee66902ab3c39c0da7866e98040e7ff00d75d17843c386c6e16eaf95e3973928f1fdd5c0ea0fd07e74a9c741ce4751a64365a725ddfcc595978e010a303e50c7273ff00eaae1f57d486af3c96d80a8b9380dc0239cd745e30d5a2b7d31a0b69d249643b7873b941ebc11d3eb5cc78674c92f66977796ab1c477bc99c64f407f5ab7b992ee5a96ded67d25a10489047be239db961e9f9715b9e0af17bf8934d6d2753f30dcc61e486675cf9c808073fed2961cf70477cd737acf86351b88a002fad3f76c0ac80952bd7b63fad75de1bb08bc35e1cb8b8597cfb894ee694f1903f847a0ce4fe35b294392c6694b98bbadf9b713436f06e660c8831d1989c0ce6b7b59b55b689e5332aba810411ff7546016c7b9cd56b7d3e2bb11dd336c4521dd93b918381fd4f6ac0d66df5ad43c403c4365776f610c3198d9278bccdca0e795c8c9e98e4753554e2ada933669cde18d30e9514b7e9e6c51dc0b891dc966507827f5e95e47e318217f10dd9d3c2c90ae114a91ce3b63dabd763b9d4f56d1e6bf0d6b0db9263db823a71d79ef5e7ba95b4ada8b4973691b5c29ff005d0e30df5e39a99df42a0d7cce7fc37ae3585c63f84ae0e727a576f67e20b8d3951e5f9eca6232bfdd39fd2b988b4e846b2c5a2450db58803bfafb57453da25fe933436ee3cc0a4a8000248e9511767a153f33b6b47b5ba8d678a5dc0ae0f0739fc2a64dcbc2ed72c72c173fe7af7ae2fc157b2dace6cae415590654b291f376aee76e03796482493bb393f4aec83ba3966acc84ca796c45ec00a630508c669491d4a81f85298844ea1997e6cf6cf3d724d4e245fe37c827a28c0aa24cbf36dc4c6360519402bb875fa54f15bc6f26649976aff0203cfe3537916dbddfecb10404f381cf1c9ce6a01241e696489091c720818fa5219719e530ed8a0754e40ec4fa565b4e89b83b3168ce0a28cf6f4e9fd2afadfb147460b903b71c678c0cd55fb05b99cc9b44aea771c01bbbfaf19a00ae20729e59dcc3f8481d467a0e722a39e06453248c859390ac0f1c76ebebefd6b42f22b3bb8764724884654082403231df1d7159d6fa0889e3923beb9201def1f979dfdb0492703e8067be68b019cf1db467cf29753bb000282c157b0f9475a985dc63f746291c633b638ca01c74c81cd687d8218773c8db949f921550b8f4e3bf7a6c56577f34735b7971330c15eb8cf43927d7bd2b0ee416d3ccdf3c90caac4804b118e3b63f1a59ed45cc1246f11642bcfeef7b6efcbad695dda425b0e8c099393c0ebc9fcffad32dac03caa89e685dc02ab01eddc1e2a5c135a8d4ac78c7893489acf526b6f2d83b1ca9e7047af4a9f4cf0d5c39568c060305bbe3dcfa57a1f8b74cdb769315dc21f9771ea39c9fad54b59edd20f91a575c02c318e7dab82726bdd47a10d55d90595bc76717cd107900ea4102997b7db240c16305972029dd8156279e279559bce55ecace178f5c57277b74165f2f0400db86df4f7f7acb565a68eb6def5766e6c8c9c824718fad3fede85b733055c7385cff00915cbc5a8448541e598f00b6303de9b75acfef8911fcc40c67a63daa5219d2c97b1cee046e0367919c56c5bda40b1051312c793c0fcab9dd0813019a58809072a180feb5bb1a156591c7cd8e483d7e9d6aac4dc4fb2abe51e3089d724e07d0d60de581b79ccb6440c73c311f957432dd32ab3613e63855cf24fa1f6aa53b85214ec20f276a7e63ad4d86a460beb019361f314afdf46e39f5cfbd5bf0ecf34d3b48d3c8113845fe103f9f6eb553518e3bb7779001ce71c647a62b4f42482cd4492a93c82413c8c0e9915690a4ec5cd7ae16587c969224014b1214fc87fc6b9187cc31c90cb119001c6e41d3b135d178b2f4c4ba6adbb83733b00b014ff568c0927eacd8ce6b14a6db80ed38b99012aeca3e48fa70b9ebcd3942cc5195d15ac2fee745bd81e45fb4dbc5202b1bb152bee3f3ef5dabf8beda4d3e392d7734aff7d4bfdd53db3c74ae76eadad92e15d0c9e63704b018fc3fc2b3de0168e1f0832d8c21279f73dbf5ad6264c9efda5bbbb2d2ed2cc79e7217dba0aee340d1e31e1b9540c5c5c7ccbc1c2fa560787f4bfed4b90f2a929d303b81d6bd151e1b48553708933b54138cfe15d14a9df566152a5b4382b7f0eeaf73a8347747c98a2cee24f5e7b7f9ef5abaf6a1a5f87f4cb1b2bf7c07123850fb55c2e33b98f419603819e78f51d3868a4baddb59993e5e578dbdc7f2acbf14699a7ebd616f6d790acd0197691d0a9208c83d8f6e2b454e10d592ea4a7b1bd15ddbc5a04771e6249e744aea531b76e3202fa0ff003cd677f69e9a3499965f2a589ce42951997bf00d72a969aa491496a7565922b650a5a58f9dbd81c100fe950683a2596a324b7fa84725d4b1b848a4272474fba3a0ea3815b7b485fdd31e495b5345b5994c421b683c8b704844438dbee71544ca02b330dee7f8b1fe35a97ba7b7d96496d8e0dbcbb2554ee3d7f95533044621b973dcb7424573caf266d1b2393beb88c6f7f2a4dcc725b776f4c0e2b3a0d4ee60b8431636673f29ebef5d06aba77991931a291d58ff002aac9a57976092754538c93d6b95a77364d58bfa55e34baad939182d20e33c64b01fd6bd156336fbc48ac4eecae075fc4d797c12ecbc89d186612801c6390c0d7aaddc0268d7ce246d1c671d7f1aeba3b1854dccb9e6795cac83009e8dfc3e950329452afd08e8783e9d6b42386de10e9701245e30a7e6a31e5fdd557849f9806248cf02b6322ac31c71c4aaabfb9518418c0fa0a410932484feed09f94e071efcd682476e23545e0a850377278e94d368acc58312a7a13f876f4a341958d840f2abcdb58c6db97803071ce3f0a959e386262809c1cf3c71fd6925b07b88e4513b61b820f07dfa542f6d701011b5f1e8d8cfb74c0fc6860ac208e1821f99ca85e060e7e9ce303e94e8de5946d89df7679c923a75f4a966b928762ce2220679e477edeb50a431862ab700b37055531f4feb4c44e20942a969b0f8cf627afb0a8aef67959324996207ca707d3ffaf51fd8fcb72d1dcc8fb40e3763f5c93f9d1e73b3b2ac6a8083cfa75e7dfa52027b7b68242408cb463003303f3639ebfe15a3a4da87bc2a8a42c437027a7d3af158b24b711cabe5cb0aa3e729b4fe3f8fe55b3a13a8b994acbb8c8b920801871d0d0061f88d54c7245b812d9efdeb839124857f7b732fd9830016418c9fa77aebfc54ed1dc6fc92ca4f1c7535c7eb17323582aca4120f2157a7e39af2aa7c67a54dfba4d70b24f6f9b64c5b8009dd18f31fb64b1edf857297f6eacacc06c743ce3b63ff00d55b56e671607732b7a853f7467a7e59aa18590949328ac48fba78cff3ff00ebd081b30ecedeea7259439e739c753eb5b36f6ef1deee9c5c6f5c000f1f97eb5d6786f432f16ef2fef1c75e07b8fc7f9d6e5c78726995b1229763f74a9fcab4e494b54887552d0c2d3354b43855c71ced2847e2c4f7ad492fe0c16da0038da73fcb35ccea7e1bbdb39b7794c08271827ad5178654da1a320a7003724d62d389a292674de7f9ea4a3e42f40339ace96e9f73291b59ba679e3d4f3d2b3236789882aaa87807f8bf9f34e866791885f30a0e14e178cfa52b94899ee0b1648d1493c97236a8fa55ab0b9fb346a85c9c9392ddfdf07dff009d576b68e5607ca07070373700773488a2390ca20423e50727d4f41f87f4a689659d4ada5bf4f32362b31755591c8fdda9ea493df04e3eb4b0dbc36fa55b2b28006480e71d3dbf5a74f2e2ee12517cc30bb818c8de381c566a5d497564d0b7fac1212718c9c609fc39fceb47aa2108d70256f9cb1c7217a86e73d2a9472497378b0479660f8518ee4f4ab66dcfdaa52b21538c02bc1fa0adbf07696835b49e40ac90a190100f51c0fd4d5c22dbb132925a9d9e8da6ff0064e9c90b461a5230e58e319ad6fb3247879488dc8c6c04e1aa05ba01d310be241c608c0eff0089a86edae6de64670de430c6f5258c678e48f4f7edc7ad772565a1c4db6cb8ce913492658a85e02ae7f23542598de5a4a16dca05933192a41233d71d867a55e478dd8a96c86183c74ff3ef43421c3e640b1ecdbf2f52493d69c95d31c74671da85e7d8ed2f2df3fe9133724b1c9c7615bfa45a35959c76889b1802c588e589ea7e95cf68fa57f686bc59d0b436ebba4c9e09ed9fa9fe55dda028877a2ee03e509ec38feb59535a17377665591fb1477d34803879e4dc98e0e54645634e8a1cb40e5908e8782bec455cd22ebed76312965dfe6b48e4fb93593a858cb16b0b7b0cc104adb8ab7194e064fd7b50b6d03aea32642f194278236f1e954efa7b7b0b10b85c778c9efeb5a331510b3824f75c77ac69b4a96f6e1ae243b8038c7b7f8d6724fa149f732acdde408369c999471ee6bd7652c36eef9b07386ea07b7ad79f5adbaa5e5a5b05064f3e32a077f981fe42bd11e2691c3e73819ebfcaae8a688a8eeca51c524d32e5484dc77296c6476cff00853830b30d24c7119c00b9e473e9daae0548d488e00339ebfcff001aad736609691495254f3df1d78adccc8d6457cf96f852d9242f1d3eb93d2a566dc008d9b0a413ce33fcce3350c566536044de33939279ff001ab0890c6c4aa12c0e7e7ec7da818e591523dfe667b6d038fa62a95c4b21cb2be4e47dd03afa7eb5633f686644c332f1ea3d7fa1a7a408912eed8a4719232303dfb9c5202a88a180ca8c5c798d9deca189edf8f4a42606915c333156e5481c67fc69639ed3c802479b38f419ff00eb53d163b91ba0690c4c32142e07d73f9714c4556b29f734d0ceac8c7e71212303f2fd29d676c63959849e71c8007200f53ea7f4a77ee106e1bc3b7df94be58f5e0fbff2e7d2a7ddba1490c00a30e309f337a8e39e948624cb3cd13c8020886727d7f99a9744b9664594205ce431285791c1ce79aad3cb733246bb5b716d80a2f29c63a74ebfe71566d848a4867c90a37212188f7cf4f5a6c0c6f16028b33ae096e178c579b6a3248d68d296232e14718e9fcebd3fc4199a021b0c4f0aae3233ea6bcd2f912496e6146c28395f703a9af32a2f799df49fba2e9d79b34a6463b9f7739e075fff005d528bf7f748bfc258281cd5754922b52361c370a4f4ad9f08e9735debb6c92ae555b79fa0e694536ec54da4ae7ab58ac1676b1410a65f0327a638fd6ad5d76e09c9c7a7e7e9549c2a4ce61b5572bc1f293a9f72381dff00c9a6b2cd3cc728e823c8da5882ddb181fcebd34925a1e7b7722910b1da6480c67395552c73d396247f2acebad22d2e1542632ff741eadf4ad3be5ba82cfccb7b5fb5361408376d6f7f98fb7ffaea5b69d0422e67b7581b67ceac012b8f56f41cd44a09ee5464d1c5dd680a01001418e38acc7d159070463e95e8b2de59dd5a078c6f52b905d0a91c7b8e0fb75aa83498ef80922ca2119cb1e9584b0e9ec6b1acd6e702229212c1c92707eef7351388502b06018773eb5da5cf8718c87e70dd81e9fa567dc7842e0e0087e63c824e6b1741a34f6a99cfdd1dd13dc447253e50df87351e8b643ecf24926efde13b5bd8d5f7d35c5b4b64a30d1bfce71c7d2a949722dedf0a71b7804fe5536b149dd0c9e78adae36a151ba3e481df3d2bb4f05811d94b77223306611a003938e5bf1af3340d3dd45122ef91c8550bce49e00af69d2ac4e97a4c5686324c5ce57182dd7d7af35d1456b732acf42d4f04170859a36cc6c194b020823a11f9d546be9255f2f98d48e78cf4febef4be7de7da0c62d8459f41b81cd3bcb88ca470a4f270338aeab9ce914a5b75da65883c408ea9f29073dbebf4abf70b2c162647fdee4aa8dbc962c4283cfa120e2a2934f96689c81f29e30cdc103b73f8fb54f6d6ced24704f3c59691711c672477c7e14b70d8c6d0828d53591112b6e970107f78633dc7b55fb9d4592ed6244c47c127bfe06b3ad615b4d675bb76c2869d581dc470476c53a536ea37ab16ce48f4271d738f6a525ca0b533348bd8eda6923901db96d809c61771e3f2c557b692eb59d624489b2bf79e5c70a3a703f95646a2e25943a83e87d4135d8689650e9fa7046c34f20df21c67f0cfb5610bc9d8da565a949bc33a808e77176b22827e53905876c1f5f6acfb2ba65fdd80cac41c1cfe55d54b3c841072001d7d2b8bd51bc832a47f2b0240e39aa9a70d51307cda3363458e3b8f11cf7ccbba2b38f6a8033990ff0080cfe75d5cfa89208da6351c6e0470315c67842656b09616471fbf24cb838fba3a91d2ba2125b34cc92ed9011ca94c8fcab586d7225a325172d3b6318dcdf36def9f4ab490cb2392c1c051f77239355a1bbf9bcc4594c83801b81fa1e95715e62195d4a64752dd79f4aa25161a4110000f9dfa006a09a260c8d962339d9d467fc834df2d9641315ded8c60b71f97ad34dc3b1c34807aae3071d3f0a2c023493ac602939c8e8a4134d793abb80d9e727b02453d5a31f3a14058e4be47cfc7ebf5a729568c2e010fc80dfae7d7f13da980f5b3b6b5806f88850877b36587f8f355e3beb17cbc32a34393c296ce7dc6322a1864613470b466452b80c7a6476f7ea79f639ab53c16f31c3c7193ce54e3693c75c8c9e94802394c8125638fd4b67b71d3d68ba9de1c29dfbe4caa92e3af6193dea35b68226688c8c180db8c93efdc73faff003a76e6370b9769323ab0181f414c081fce55611b04c81b517e50c4e7b63dc9a9618e711a2b3227cb82a3a71e9e952e4f94c2089371cfca78cfe1ebce2a28d24645df80fc072b26429fa9e0fa9140195e23b39869123c809c02719038cff85723a3d8adc481a44189176e4f40a3dbd49af41d4edd4e93387dbb5936c6a064e7bd72b6768f6f107645455036827dab8ab2f78eba4ed139cd7a70d770da5bfdd5383b40fcbf3e3f0aeafc19631a8bbbf9c8223222427bfafd7b0ae41ade6d435d5b6b701d99f0a17b9f5ff3e95ea56f0c7a4e9f0d8c6048147cf8ea5bbb1f6cfe5c555086b726acb4b16249706201a46180aaa00504ff009155d8b4bc46222cb84e3b0cfaf720fbf6a91609b63311b77633b4962303a7bfd69b70c0fc91ca09504614e3278ebd81eff5c5751cc549a18c27ef6e631e4e0bc8dc0385dc73d8f1f4e9d6a5f21c6d963b9d90901d782db89f53dbb545f33023898b1f26456e5073e9dcf6a98c52ab14b78d51130bb4a724fb127a5202bb59c734d148d713821b3e5af0243efdff00c9ad3463989408dd558a484bed29c70401f78e703f1f6aa691dc2c5e6f9d928725771cb8ee2ada7df259c0ea0283f363e98f7a065767862b992468cb3646cc0276fa66ad2131fef2695107604ed04f6e7350dc7ef20478a32adc150c76103d7d8e39c542b1e6708670f9ce4601fd3a9c7f9143031bc420c42e2789090c43394ea07ad7189631dcc4fbd81dcd900f04f7e3dabd3afe348c84c06122322e3d7ad7992b3417374ccfb9796233d083fe15cb888da57ee74d16dc6c5af05d940fe3485dfee5bc6f303d70c3819fc5bf4af5392e41076b7e3d2bccbe1f06b8f125e4a87605b661d3b332f1fa57a1c8e911decbb9946ec9ebf88c6735a51d226555ea45777d3c64019238ea7ae4d661bd9a63bd4ba0524376e7be3f3e9534f73be472e0855c95c71c0f527e94e8635bb4dd1c8d1499dd871d3ebc75ad4cec565bdb968bcaf378000dc54927009c027ffd75abe19b4371ac6f322a88a3672e0e79e00fe7fa567b69b237cd3440ec24a9539ddef9a4811ed5c3452ddc45ca6f0854ee50cadb324606718c8f7c73cd0b7d41eab417c63a5be871c9aa8b85944856378f1d7e620fd482467eb5c75aeaf35eea56b1918872c78c1c9d871fd6bacd56dae7c47770cf3c40dbc6f2c82252555b7952b93e836f61925bb01ccb168f0da46008a24272536af3d0f393ed9a753508687091299ee5514962ce0f1d4d6c5c78826922c4798d49e063b7d6b26c6259f50fb333346a410486c10003d0f6ae927d25563df850148e00ce0678f706b0a4b4349bbb3146ab7c73891f71e3e6ea7daa7b95de1e57e72bb89239e95a565a233c1bde2608c49c0046475c9cfe14cd4ad4db69f745b82a31c7d2ae6ae8986e666853182d1c2332c92393903ae3ff00d46b5e3b879305cb6491b593f539e950e9367141a4db4b7184df1e41ea493cd5c1b445ba157703bedc679eb570d22852d58452c99013aeee70ac47d78eb57c4ee91a977c81ce17923d3158a66637223786553fdf2a42fd73ebd7f3ad55111561bf7ee3ce38ab2470d4d965770728146edca3f3fa539b50925fddc71be58f71d3f4aaed108d4fcc72393c76f6156eda5923255610c010776ec0ebfaf14012c0c662ebb9978cb32f5cfe3d2ad4717923cc1282067397e3151c6a9bda4685429e4f2707ea298f7b86dab6f1941dc8c8c75148099e588aa955da5b1d24c73e99cfe18a89af1da4db1c242aaf395c31e79e49fd4522cd2c91c612cd90a80553675e39cb73b463f3ed56a2089b9a2b65f31bd130481ecd834c0ab04e2ea7321e5792158638e98ebc7e35248b721099218d5f394dec39c7b91d6acc2143079195243b810b800fe1eb50cb7492cde520666ecc324904fa7af1fa5218e776448e56910c04659cb7207b1e38e3ad2c276466473bd994aaa2752476049ff0cd3f6c4db6499501230bbc13b57e9d2a9cbabc505efd9e3726775053cd8f8000e80ff11ebd4f7a0445771cff006a9a49588b58ad54a47804f98ec40fd01ff22b9ed577e9fa62daa92d712b12c7a919ff002056f5d5cc979aac1632fee6ce284ddddc8ddd4640e7f0c7e7593a33c7aa6af73ad5cc6df675c18d141c866385c0f60335cce3cd2374f9625bf0bf869b48b55bcb8205ecdd037544f407fbc7bfa74f5ad379218a2ca05085891dd81e9f2f3d393fe79a9639e394fc9012a0e376ece38f6e3b735595e5b83b8cc304828b145907d73c7f9e6b65149591936dbbb263a9f1e4a3217c73122ee603f3e3d327dea4fb296452eaf197c123ef1fc71c76a6bc91db265e2059c7088c0678ef508d52e65489610a37a03c1cb28f5393fa7356845847b7850a6e2ae0648d8464e3b8fc3d6986712c98f2e4258640ce08e39fe755ae1d6231b87b86720ab161f7b8e8411db3f5eb4a934924ac5d4074fbb2051d3a73c0fc87a520274309b801199a48d4a95690285c9e73c75c55a9a32158961cae141f4f6c0acffb64ca31b40638dca8791fe24d2437a0b0621811952a4e037ae33d4ff81e949016d72c8d18951491ce7192303bff00f5ea2b6f2c798980ac84e19e4c9917ae739f5fcaaa4b265143004124e092768a218da23be46577c6d2c000b9cfe9d68602dfcaf6f6f14b260471c8b80063ea7f5ae0f538ca35db03864623a57737317da3499e1219000484fe103d45729aeda347791923f77756a251fef0e08ac2bc5b499b5195ae87fc2f5097faac8e7016288138cf1b9c9fe55dd5e457324a24802c80b0c83d4f6c1fceb86f006c8f54d5d5b6f1044f9cf030cdfe35dbc93471206f382a1c6d2c7031d319fc7a55d3f848a9ac881ad4cd3289311eecf98ac720f3f9d5a22d627528410792ab92c7f33c5539365c49872ab21c819e303f0ce7afa8ef5198e5b6e61647563c92d838f6f53edd2b4b105b9751dfbe18e368c6ddc1460b63b1209a5b747662f218e40395238e9cf735022bb1dac14003a13c83fe7bf4ab71e1d94e4004f27db1edef4ae0493ab4bc35c4ca00e81b19ebe9fe7d3a544ca5627df334a029c96c7e1de9ecdf2f2fc63a8eff00e79aa379f2c5238ced552ca14704e2804717e1b74ff84b22f307ca4c9ffa09aef1efed91c98638d9c8c861d715e75a7c132f8b6ca342c0994ee23ae3073fa66bd0046239c223ed5e719407db83ea49ef9cfb5453d8b9ee4b16a26793caf29bcd1c95c7538ce073ff00ebe6b03c592797a55c7c85777a919e3e95d12cf0dbcc91a80cc79c16c31f538ee3b67deb89f184f3bb98198b8418662a14313cf03d00e3fae6aa7f0b143734556e64d32282dd911e38c4597c8008519cf7efdab322fb6da6f7bd955b672de4167e3e9838fa76f535b46e65b98c7d9dd0a10324fa0ea3fce6a94d75bf0cacd8272a7b75ebc552d096466f1161cb3056ddb1724fea3d696cee9d16469e6f346f27704dbb47a71d6aac973ba405b686f53825b8aae6e26fbcb85238000e9ee2a908e8edae0ef2c240ca065573f81cf3ea2aca4abbb72c230a48e47d3a76ef5ca5b5c6d9ca6dc150586d6393f55e722b62d6f3076609c72bf2e3f1e783dfa74c517036cbbaa9c9e78e17827dbae2a4743c32edd83ef6e38c0fad657da4249e51594b9c90c559973ec4f4ebdba55c8a70ea786e46d6193f38f4f4c7ff005e988d50f711c6d124aecce0ed91930074ebea7af3fa522a370af2ed7ef9c9247a8fce9e218e46572c4b704e380719fd064d25c43711464c449989e5b2001f9f41e828432b4abf29dc1b18c960327e9ee78c5352078c334593b73962a415e7ee8e3271552ee1d4a28239239a5de5d4911e5987cd920fa03d2ae4ba75e302b25d001cf3bc16c275c6d0719cfe9537d407246e6390ab860a005972a4f3838e0741c536d605b80249936a0cb23e0107f0238cd5e61080cadf3be30c0e7071ede9519921fbabe584270c547191d0d3608e63c5f148e638609844b2c05188e0b007a01e9fe152e8aab69a3e2475702422440d8298e338efc01f8568ea16b1dd5cda330dc55d95b70e8083d3d2b346876cdaadca5c2c8e301c2c6f8041e704679a884356ca93ba45f8f54b6113a5a441d57828b80a493eb8e4fb5675e6b7a9465d64851ed8afcc88395cfae3b76f5eb57cb46266857642aa328070147ae31dcf7a8cee4708434878dcf9007d073fe3542313ed57ba95c25d2c6102723600420cfbf1d3eb57a7bd104aad26d328e54b107b75cfe38e29979a98dbe5862a3d5482077c67ebfcea8796f72e19589c03b958800e3f97bfe14ae04eda9dc72cf216888c15da029fc3927b77ec2ae4378cd1e42ee38c0506b217e556964963909c9217819e80f53803a60d5879d48554c9e796ea724743e9401721bb70482df31197cf3803afe27919fa5569aede4b8d8243171b801b4ed1e83d0f38e2a386cc5dba80a5c0c30e4819ed568699fbb58f603233300bd4e3a9e3bfd68b8752bc378f14a42b216c777c363dc75f5e6ac6d6780c9f687213853d73efd3f5a8dac1e293710370c924f5ff00eb77eb5249179e8065c853d8f3ee293431d63ad98a49a096d6e92dd484064c1de0f71df039ce79a935787ed1a6d9cad130786e9a22403d1b2463f035992a243b63540a598e001824fb67ad6fe8b70750b17d1dc83231df19072e18647d318ff0a76e65ca24f95a671fe1e91ac759bc608df344aa147a86f5f6ebf8574526a5b8e31b973d08e9596ba4c89a9ddc9793bdac782159b80483c29f4ce6a4df15ba6126c7bf5cfd735118b8a2a52bb2d9d52411c6000b82704e0e07b7a527da1e4257cd70c0e700e3e98ac9170b2805645da1be653d71f5fad38c92c7800867ec03631fcfdbf2aa6c46c2df9451fc2cc474e4607d3f0a4b8d46e884c4922c59e5b3b73fe7fa563893219022a8072cca739f403d79cd5cb4b8322a863829f291fe3f4a00d0b1ba9a67dc59cc6064b33360fe7c64fa54a97bf6cba16f80448c149ce33ebfa0acc68669d364624c2b67046d41cfafff00aeb77421169fa80b8ba1f2c6bb23444cf24e0b127d067eb9f6a71d5ea27a233358863d3753b5ba8d24386db955c900f19c7e34e93594fb3ef562aeddd9b715ff0080d745aa1b0d66c27b768595a45740546d2877655bf1033f8d736da5e96a196092403b6d19cff8d54d24f4262dbdc1757967c6df94740c78ddeb9f6abda858437ba199c48ac5223b885e037423dab2d34f891f799647cfcaa17e503d78cf3d2bb4f0ec56975e1c4b7948dcd33a12831c92580fcaaa1152d18a6dc55d1c1b5c4b14223792e3600070848ed8e00e7ea3f1aa8da9d922984dd44a42e769946f07dfbad745ae45a7db6b16ad625e35915d5d0742cad83d7a1ea0fe06a111c2d394db0b1c703683c526acec52775739959d1fe581e3931cb15704e3ea29f23e2277000ea09cf7c77c74ae97fb3edd226b85b5542793c727eb4c11043b400acc792703f3f6a4073f148ff740f940e0839cf3daae45364820ee6fc0f6f7ab13d81daec6dd5a23f7b00703de9b6505bb392226753839906029f604723bfe34ac08d1b65925457460aa3a17e377d3d2b49600519c4858f7e7fce2a840311aa274078da38f7c735a1044ce42862c323af18f7a60cd09ee3cb972aa02b7cacc0e0003b9f5a7c6ad364e5f049396270bc5321b052637f34311c6dda0afb83f5e327d862ad9b795652db932dd704e71e98fc7f95005525398f6aba31f9893ce7dfdbfc9aa76da85a4b611cf6f22c9048e42c902eef3704804601e3be7a735760818210e5c82dc2601239efdb1d3a54b2450c6c6354442ea490011cf5c103ffadd68604170ad24441dfb91431da39fa1c9c7e7508b7636fba48cb1655cb27de23d3a0e3ffd753dc5c470ce03ba48a060284c11c67923f3e951863f658b6841348df203c96fc4f6fae68193451664889d99594003a16e79ebf535b2f69a3b5fcb6725b0798c1e7f9cc7010648c6e1ca9eb587e513b54bab485492338e7d063af269c22f3150b2375f990b8dc3dc8e94e2ec4b57305b4ab682e273f6cba6532b011bc83279c104f7c1e33ce463d6ab4d6685842164da33feadc301d0727391dff015d15d44cb084899233dc11d476c1aacd6de4210332be005db1019e3b91e9c54b1ad0e4eef46b8b1092c44c88390edf367f2aa46eee0208d4103850107e5c7615e8522e6354915783b1970739ef8f5ac9bcd0609db7c811579032e3047f86052b0ce66da06925794ac85385cb202073d9baf5faf4ad3b4d26495497e092597031cf6ff003df9ad98aca281961f3230e1436003c83dc63a7f9f435a185811947d17b9069815acf4f86ca022404b639cf393f534e2238a32c32d824e1475ff0038e94f58cc85b3b89e849c03c7b7e752155b4883ba91b7070847e0067b74a40431da348c4c9b4139ed8073ed9e9da9e74f8a44f99700363711d082455192faeeea50aaeb0ab3ff000af3b71ebf5e7f2ab10823e667918918572f9ec280092c2d76aab10ddb23a9e71d69f0edb1b9825445c2b05eb82067071f85442299d98292aa0804f5eddb1d6a1940b5946e67793b741ffeac509d8373a1d53c3ebab5b4823bb8539c96232b8ec6bcfb5cf086a7a3480c8b0cf049c896d98ffe3c3b67af7ae8e3d4ae2d55a38be56ce4856fba7ea7f5a7c66e9ed52d521792143b54349918072339e78e3a76e315a4a4a44462e27096d025ba92038218b333124839ce79ab968fe6a750e476ea319e9c1ada974f996e87ee64c16c108b9047e1dbfc7bd5bb7d2a4dc245b768db6f25fe5ce3b56762cce540bf7e345e7838e9e99ff3f95588c2bc88ca082fc0da0638f7ebdbf1cd5d7d298dcb8f3106e193cf23d062acc16296d29ccbbc11dbad1601b1580910704e7eeee19ff3f5a9d2cbe60594939f954fad4d23129b62254b7191818f619fcf8aa972b23811ef704e4b0071d88c7bd3b0105ca1368b1c47280fcd8fe559ecc44450fca594ae03f18ef8ad0b69445fba9301401ce02fd00fe78e6a49ad03e30558b300c30720fa71fca8b019d6cc3cdda1176741d46463d31c7f9f7abd0c11c7345710a059524ddf7b1f37233c7d4f38a85ed67b4db912483772cabcfb0c01d2b5e2b57c72bfbc18c6e042138ede8073db9a15d068673e8704f3c974c24f324732e59c90ac400481d06401eddeac35a8b0b72f1c6647ee7d6b4f31c6eabb8b313c003b7a7ffaeaaddce766003d3039ce6988a7736d248536cc55a418383dfaff00f5ab38e9d3eed8f6eff29c865f987e26afd8dcdd4f71709716917951f10b2c858c831dc63e5e6b48b4a1b2485e33b739dbebde86073d6f09657fdd146dca195a33827f1ad2b6b52a3032a319e9f2a9eb565fce64ddbf664f009273fcaa0779963513b2a871f733b8e70723d29017027941433a8e4671d48fa55432b35d98e34dbb71b89391fcfe9cd5786e64d92799b86180cbb13c63ae79a65adccd2aa6551fef30961f9918fa678ed54c691b714b1c08809757662768192a33dffc735107bab89d884261e7e7c8da31ebdff002a31f68bc95a28a47518c197e4000f61827f3f4a7cb6c96962aa079880ed5466daa4761d3a7b5211544a904cce8a58ed2a647c0033fdd3d854f0c6f72166599d1b00a9d99c823dfb7ebc5526bc68de455911644c00c1549c63763f0fe9d6896f2e4ab8dd336e3c28c65b8ea077a40b5279a58d418412c4023cd73b7b8e8076f73e956ada2611a2cac9bdb8055fa0ea00fea6a2b2b62765c3b326d1955900071ebfa55b91d95d95e4f908c83c023d71dce38edde9a06226d8806725f8e368c914d9fca512481082e5497cf381d00c74c7b7ad324b88e3c3c92c51041852401cf3c7e9cd31585d44c5446d938023e07b73d323bfd2802022212a4a102cbc90864e0a8ea707dff009d4b12b2b16997cc76c6c6ef93fc874e94f7984284f1b89e0000927d78e9fe7e94c865774ddb8fdec6f006474fe540131590a9699fcb55043038c1c77c7351798318463838dc5b19f5c8ebee39a6bb1f319f73328396e7273c718a6aab392af9c29387507a77c11fe140124922bb13fdd20601e07d78f7e94e3229c08c33924b1d83fcf34d8d4ec2c06371196552bf8f34f8a4730e0b70c7380303bf53fe7a520226ca9080e02f04e7924ff9fd2a19312cbb243ba3519249c631d3f53534cad2a84c67a71907033fcaa38ede590bec0dbb033907391dba7a8a760293d93637c586665dcbf5c739a746b2b4b1c03e69304619b383dbb63ffd78ab33e9f22488448188190ac0edce300e01e78e39a00bf8632a2399c93d547f5edf4fca95864da8cdf648046982f9f9829e7a66991dac33461e45452c72703927d4d574d2ee6e26669a5752a0fcb8008ebc7a77ad4f2628e30642b943d3a81c7614580cf1a445212772b066e3110ff1c7afad5e7f22cb6c5bd1778f923079623a9a59ee9230444db3b923938f607b551943336e2ec0e3a0c0c0079c93fcf8fc69340586b908770f98f1951ce4f6ff003ed4e6dd3965e5495c161c7e47d6b3a0dbf6950c1821386393cf5fe7827f1ad8822ca0f94a1e803939e3814c085f28093b8100b1d8371fcb1cd54b3be177189d74fd4625eabe7daecc8fa1391f8806b45c14cec0bb48e40eb9e8463d3deabc9249f287954c6e79720623001e9ce07a53100de12632c6eae0e73bb218f5e39181ce39159eea5b6b2ed1839c6dedc0fc2b4f67ee2642aa09ce58e39cf5c63b707df8acc91be50bced3c1c679fa67fc8a008e08d229239d419641ca297f5fc71d3a67d2acec0e222495207ce1b239f63ebeff004a4b5510dcc2ac5f07800a6e2b9f4e71c81572f229e1756b748980e082fb700f39fcf3c7bd03248e54ded1a87f918ae4e483c03dba8c1c67d69b24c554b6d2e3a1da39fd3fcff3a162981cb4f1b10c30083cf7eb4b2472c991821811fc630467b7ff00aa8b0990e49e436c1c3134a238ae00524c80039238c7e7524b62177e10212492cc7771ebcfff00aa9d046628f6955238c61b049f5e7140145a296d5d0c298c28ca1232a0fe956c0dc379ca8cf2a33fe1eb524a42edc6d7e3e72ad93803deaaac23cc2ce19881c6e27a67f5fc3d68b012b3c8b9c00ced9c9ee7f0f4ff000acd9164dcea70bea49e40ef5a79da7240503fbc71819f4aa856dae3f76aa19baf4ea739e9fad080cd86e4c5096117c8a1b604c9271e83b7e3c55ab74b59ee966809495f975fba49c7dd61dfebf97bd8582603e7844acdc33a30ebf43ed8a9a18307c968c8dd9c8971bb6fff00acff002a2c05883c8b15558642ec4fcdbfa903b01d80a64b74662c3e6040db8c7ddf7c7a62a1d6bfd6ff00df5fc854f7bfea4fd05005048945cb9c12b275054b1381c024d5a86d9cc8acb11dbb8104f3b78ea33d4f3ef5027fc7ac5ff5cdeba23feaff00e023f90a00a6228a0db9dc1502e1d892481c727f4a8dae4492288f0dbd485600f238ebeb44dff1e917d7fc2887fe3c7fe067f9b5360432bc69b8b1562b9cb118c13d7fc8cd2a98c011c21d588c972c5b031df3f87154bfe5e87fd71ffd9d6a2d2fee43f8ff00e8429017d637232ecaec7048ce319e993f9f351a47e7479555c004138c1e3ebf4352c5ff001f4ffee2ff00e84d5237fc7a43fef7feca6802b476ccd2e1c92dff004cdf839007a75c53d6340bb0a12131c1c904e38e4f53ef574ffa8ff8137f5a917a7fc0ff00a500674fba38bcd81502ff001ee20a8c0e7ae38040e9515bfda3c92649773b37dd88efc0efc8e9d3a7eb5ac9ff001e86b964ff008fb7ff007cd2633a2898c71fcca501e486209fae41aaf73785242b800e3202b727db9e07e34973ff001eb2ff00bc3fa530ff00c7d37fbc3f9534031a6668b7b71800bed38cf3f5ff00f5d385c36f0eb27ce4ee0bdd476efdf069b73ff1e83e83ff0042a8e3fbc7fe05fd2908923761238321c03fc40e79f43f53da9dc905938c3f048e871c54573f726ff747f5a7a7fc7c27fd7bb7fe8542182c61e6cb3f2719e3b7f9fe74e8eda5f3dcbb7c9d55547e393c557d1bfe41d1ff00d715ab5ff2f2bff5dcff0023400b0c4be6b4a5372a852a00c0fafbd5a653246e158e769181ce3f0ef543fe6297bfe7b9abb63fea25ff00817f3a622668b6db9705546d03e6e001fe7b555dafcef8c2c5cb61471c1efec4638faf5a7df7fa91fefa7f3155ee7fd45cff00d744fe6b4015aeae5df70542a8d8c280c1b3d8f1d39c0e0536368e3b7d902c79da55493d3a038c72ddfd7356dffd65bffd753fccd65a7fc79a7d28034b6ac89e6c5230039f9727ea79fa7415695d836d9086c0ddca8191fd71fe7359d17fcbb7fd7c2ff2adb3d63fa1fe74015527b6791cc2c8481bb3ce403de9cf24d293e512b83b781fd7af439a597fd645feff00f434ebce8dff005cdbf9ad0320dac028dea4ff00b6c48fc7f5fc7151dc4452d1a4cbe40ce117803f3a7d97fc84aebfdd8ff9b54d37fab1f53408c085e54466556049e379ea33df1edfceafc73a07c3c25832ee2ca7206307a75cfe159edf762fa9a659ff00a987fdc6fe7401aac9e71de1d8a30c7963231f81a72c61a1f2237cae3866c7cc0f63e829f1fdd1feead320ff005aff00f5ccff00e826840cba51208d628954e0e31d0fd3ff00d7514ec6180b2af9800e1700fe008ff0ab16ff00ea23ff0073fa1aae3fd68ffae6b4c47fffd9"
        image2_hex = "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909090c0b0c180d0d1832211c213232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232323232ffc0001108010e019003012200021101031101ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f3ef298aa120a96c77c7e23d8f1567ca9157cd8c2b007207a77fc0d318ec1e5ee2e88bf2fd71d6a65894c00c80e0421be538383d3f5c573b39d8a233e7c914de583f7436e1f360f39c55ab68e3792349832b20dcb2274c83d0fd4f43fa5507625e4dca448ab80412771cff003e2b4ace46374913280e46463b0c918cff00c085449b25a6848fca8596156224396193b4a1183faf5eb55a30558a48a44723129b8e09207f9faf3e951aa3496eb8de1c9e4b6304e3b63db15618192d92471960c41f40c14631ed83dbd2a5ea68ba96fcd79a3450aab2a600f93ae3b7a54461cc51a5c480e092406e49381f5a44f3962374588ddcae7eee41031ed8a8a22c1c3b6e53bb0a4f7cff00faa84ac5c2492bb44e2288c9b849b572471919e7b13dea586f6101ad8e42919523236f2338fd2b392e1a08c3b0791baa1271b173c018ebfe7ad207ca60928149dc3ae7231fd4fe557ca69745f378c0068df6e0f53c704743efefef4e49ccb0491c52619c80e00eb8e715972bfee8aaee0accac0a8c8e9cf3fe7a531649725d3920039c7278f6f6a971266b52f2b0652242c49192477006300fb62ae5ab2c71c90a10cace582919f9b83c1ec781f9565c72a334b24780c319cb64373fd29d801a4f3305b2410adef9ebdbb5515cf7d2c596612306902e58103e6ea33fe7f2a48c48aec09dca724155db8f6c7e74f50c8c8aea1958eedbf74e40e4f19c7ff5eacc6970242d133065c32a96fbbdfb8c1e6a5cb4d4778b5a94e4b6510246e1e458dcfd7e6cb60f3d29628432468b11c6df937b9f6e08add781a7065888c9c36c8fe60adfddfa66ab279b94923478e4230438380738048ebcfd79a95365c7c8cc9ed1d0808aa039f976fafe359ef65330333c5b235386c364719efdabaeb9d3e6255ae203049e582cec08efd7691900d549ade39e262721622db502023dcf2724fd69f3dc994548e54c1e5ab1c34687a3aaee6504e300fa9f5e82911097650ac9276dc0e47d7dfeb5d3cda7f970abc7771cf36c1e5dbb26d71d49fbcd81db8f6e3ad73f269f30f991658e6da03f97d79ea7d4f3dcf5a69a6612a6d7419e6798c1770e0108cca081ec73dbafe553e591f7b2892318e002793e8054f2d999e468b3990326f655e08ea0e3b64139f7a6dc086d8c4240cd212730a364b7230411d303d8e69a772631d6cc478cc9a9c4bbf7038d8dddf239e3fad4f612ee791bcb562b00628c787c63afd40150c514e4890232c91c81c1ea4aeecb2f3d0679f5eb56a083ca48d1d420ddca9e0019ee3e949ec13d0b2e8c6e6548a758d7218720e38071e9d09fa54a67c10a1079bb7e504f2003d08eff004edeb59d2796621185c718f978de3b738fa669d71b6e630f229f35a2c280d900f1ca9eb8c8edcf7e33516b93195b42cc0c842edf95fcc0393807d319e73d3f0f7ab36e449aabc48a6497cd2429079453cb1f403b93d323d6a18a169278596294e1c48ed8c018e79f7ebfd68b8b99d236b7b19d5559b323160acdd0e319cedee477fc8542dec68e4d9a1234134b2f94648d178591c924a93c64f7c63ffadcd4d6fe4ca42ee6721845f2ae4609c6efc391f5154a1448ed44c1f3b48c1624062720679f5fcfb9a92ce3786531c8e44c431241e40032188edd4543d187349a245c4d39fb54cc8af900c9197dca7bfcb9eb91c558f2f7a412dbdc069046bbdc6e08e4606429e467d09fc6a0851ae05bf9b09f243b2ac84a7ccb9e0100ee18c01cfa75e94b7114b1dfa2af0847ccdb0ed3ec38cf5eded426d1d14a6d6acb0630644fb5dc21c296520600c727a7031e9c915466126f9242aaa9c2c681f39c9f41c7e19a7c6b25b2ca5e38cc6ca724e063b92074c9c1e9fd292236f315731c655db71da9f786d2140c9c03d7b55a91d909c6d6465dccac2ed42224a17929963963c018c73dfae3a1e295a4590c76f28959947df53801b3d0023a7f855aba86748a1f2660b3719c8c051f41c7d33cf1db355cc4e8115a30cd8dc4b6703e840241eb5a2d0cfdd5adcabbe77f9915495e632ea0b0ebdce45565df6cead70aa8147de751f28c75dbfc88ad251199ff007815646e17672c7db918f5a6c9fbd990a4a919c73b97257e9c0f4f6ab522ae99952348f3e2293cc24648196fc7ff00af4c958aaee7922538e76903f5f5ed5aec374062790052fb9867804f5c8ef50f916e492d3c2c4b63e71b72471c7f851ed2299939c23b98aeb6cd82dfbc7e9fba1fe269ab6c32d2c7b9fb3072060f6e95b8f02c449429b73960dc0031eb50cf179aaca23c0380e9918ce474f43f4eb57ce9a33f6d17b18c82401c8601c02c4839ff00f57e355c92cbb8a3631c36493ffd7ad25091ca76c00741f7f6e3af5e3069c605f2a4ddc363628ce3001ce063fcf068e6ba14ed6e6465858f0a4eddc3e6da5791f85298a60d9120da79e3b55a023c1c0512673bce08c7d698739c1191d769e7156a4382ba10bca9b54e1948392a320fff005aa581de4ba45452ec485000cff9351c11ab8c15c2f5dcbd7ffd54187cc73b18aa0edc924fa7ff005ead4ac3f66893cf3c8e0718041ab5148b1e5324ee4da4638e7b8fc87154426f2001862793538c46d280c0ef5d8ac476cf3f8fff005eb27d8e3516c459010ecc783d7923bd6941b6092cf1f202d1a8c75e83207b648fcaa9b1da1976f28005c8ce73ed56a58fcef21d118b9c9238e3a7ff005ff0fa54357069a0b4736ecd98d4020905c1c020e148fc4ff3a589659e09208f0f26e8caa81df38fea6826465512ef322b63e638e9fc38ec0f6f7fc691267446f2be57270154f627040cd249dc9d7725902981e1b7e52240e18721c9eac3dba63f03dea2f2a6926576f302b36555810c3007e5d78a7222f9526e4384eaabd4a64f03d391536ef36608c009f1d1d89232338fe54d58da9abee446de48e6602250700062dcfe0074a81136b1f936ab1e4aae4b639f5fe75aed6b2491a491c01caf19e48e31c1c03db271ff00eba892d24deeaedb49fbf118c7041e9c9200fce9dd6e74c28f733e1612e7e5d8817018a8217f4c54a966a243f300a7ef3377f70056a5ac36b34d2fda1becb83b6159636f2c7aee206474e0e3bd588f4f0f2a243c2e392abdfa9e7a0038f5e2a5cbb1a3a77b99034f7763e5b330248ca8cedc73ff00d7ab70e9b2ca03f1285c1cede54f39c7a9f7ada5b4302c8c4b3e3bc27686e339c10339e476e86ac5b5997b18e55591e100617682403df804639fa54730e18748c6b6b2211000c553280edc6d3d7e99c55f86c05c205965859953cb2b82188ff6476ea3afa56ecf601a28a2999da364f910118231d00e73c7b0eb55265b78227f26dc236368ce588cf6cff4aceeee4ba0a3aa339eda38cc4a1e59154861b46d0e7bf6cf7cf3d78ed4f59c88cf931e30a633b64e7afba9e78ebeb52c4259229188f953a81dc7b81e9fd2ae4496d3ac716c74791b25da40c39cf1d06cfaf3f5e69b4d1b46117a14200ee0cad14eaabbbf78d26e639fa8f50383ef50082225bc908eaa497565c0c0e0f1ebd306b423602d8a3403cc73f7f68c038c67b9ec2a0cbf945628d1234da5641b72fedb71c28f9704f3d7a52d6e733a6f9ae86c16b28884b18112464b06d9f2fb6010463e94d7b55f295e4def96dcbe73e539cf5dbd73f51f4ab6b3ceb74cc22690c6437cc4800faf23e9d8d54291dd45248b6de58524965257767a918ab46b6d0cb9e007e71329041e020000cf2a31dbfc2a3d8d1a2a1b8664006013938e83af5a925b36424c6ef9279555248e7d075f5aa8f6616562532c23da460f3eff008f3fa55a48c1efb0d955a3664dfc0eaa5b39ff003f9557b890790ca0ed2980598e73d734acbbbccc603938cfd78edec2a20bbbe488b03b463009fc68b6b731b73cae3e40e0a82e170a031dbc918ec3f1a7c7220748d9018cb01870338ec71d3350f947c91f2faee937f5edd0f4fceac2abb4580d26dc862a622de809e9907a52b0d524de85fb52fb947cc9c18c8ddf2be0024ff9e685b1f3033794c64cb9501b0ac0e481f871839ee7d053eda55fb298b2bfbb909fbb9014f18f6e4e7f1f6ab623f2624676690302157015707f88e79e307fc2a5ab5ee14e29dee108925895142b22aa290ca092c013d3dbf5a88c67cce124f30f981dbef03918cfaf73cfb53c98d110c608490ee552d9cf03af761cfd39a9c5b5cde23c91473bab01b0a1e739f4cf38e7b565b31725a44096e2e9e6c4b0c490858d8160300000f27d4ff4a5172b1452a2dccb2320050283c90338c939269f259c8241081247fbf2c59a2eb9e14f4e99a82f52117d288a6fde1195da4fca48c0247e07afad256609b5a11492c971f336f59bb47b4647a8f4fe750c8d751318a4f362db8215c105483f4cf43561b4f8ddcabdd33ec8fe6954f2c73e87be38e6ab4f673c7179c842aef2cb237cbbb3d32067ae2b4e5ea5494ed726b4d4bc9b730ddab36ec2f0c032af201c8fcf078155422981894223079551f74f427f1e7f115018ee6290bc12ae4fca06dc9c76e08e0f1fad4574678f3210c1d94fcf8ee3839c718e9c55a22eed6638af9b21b40ac7673903e6604f5cff0087d79cd3e15fb62ca8506525c01c81b48cfaf6c7ad22cceb1a48147ca76aa900ee1db1df22a59a7890aed570fb73c82c31e87db8a36358cfcc7ae9fba099a17122a7ccc22e768f43f51cf7355e68995a305fcc629c1c001bb93dce79157227010968764cbf7564cf4e9c669b6a6da68e5591c11bc0903b71d71c73c54bd4d5f24994cd898e32a032b3e4a059371932391938e47bf3827d2a17b798afcd295207c8b8c727d4fe7dab691ad10b243e4c7d41001c9fc73cf1d3ad42f345091236e1281b9818b60ce073b4f51ed9e69ad0a5461f123364d3a78825c4bbc233942aa096ce011f51938f6ee7b53274b8fb41c8448777b608ce0f3d49c55b335c13e6cb399c91b58b1c1cfae06307e9c7354ae63f35f0fe691f7b630183f874c7b569127914559155fcb8e62be6c51e38da88bbbbe18e38c719ff00f551705a1cb890649f95b19c0ed9e38cd5d7923554f316130ffcf2da54377ec7239ff3c5562f1dbc59055dba0690f4519c93d81cf15a455c7cb645212ef62246cee43bde3214b50f240f1208e1750328c1c8f6e4e381dfbfa5225ff2e1a4684c6063e42e581ef81d3a0a7b49101cdb3ac8061e39147503db83ce49f4abe512d09422e15dce475c2f5fe74e63f330d9b831003024020fafad5c81618e5567465427258004e3a6067239a375cfd95d090b93f2f3ced3d70473deb26ce352d0ac0cec599c7d739f4ff3fe454f2bec00020823033c8c81c1a5f227fba2dc9e412cb1e7200e9f99c7e3560403caf29c00c778c96fba1874c0e3ef01f4c9f5a571b778db728c4cc03850cd93c60723dc7e62a63132386da3702189c918190738ff00eb5091f948a3cd24eef9fb01df8fc39a9ede165912ddd9b804e42e55baf18f5a994b51792210a1f7319d6366076965ea7774c75ada8d8c76b869a5da46d0a0e4039e3b63a7d6abda5b47e71669700be7239eddbb83d3b54fe479ca11f2a91839279c118fc31efec284cd29c9ad121e96d34865d8d3c884794618470e4f000e327a804f4e79a8d6d41924410384420158c6fc0e46ddc4e0e3a7a568410797bda00c06d62243236ee9d71d867f3e79e94f1687cd082e8a2edfba8a5b69e07001f5f6a9e63d25d2e578b4e75b825a5850b0050ba87e3a02c33c9191ff00d6ab3259469760a34a50f11a9db80071c904024fd39c5392c5047c2caef8da0346ebfcf963ee7f4a7dbc68fb42c0d2cebc1f29937aaf4c8cf3c6719f726a6e53b2d8b08ee8a96e104231f36e23ee8e31d4f5f6fe7cd5b4bc9e660b6d3448cfc2b8858aed07195e73c93807d8f159f0a4302c605a451f006d8c8dac793db3ce4924e392689afda37c8da439e58f3b82f002f1c63fad2ea4b66bc9621e66967baf3d5176a6e5f2cf5fe1c1e7f2ac8d4648edad65b8110528bb109c67731e7fa55d935580309a7ba45206e5dab88d87519e73c7009e871d2b2b552b79a5c278c380e36fb1ff00ebd5d28fbc4cacf433adb509a01be1271c86cd6b58dd06f2999da30c0bab16ea3baf4ce0726b1254485e35826f389196503ee9ad2822f29bcb90900062b86c7be38edcd6952ce2138b8ec6a3910c877848c85072241f5febfa7348b776ccacab30865da42a3c6d231cf00eec81cf4f5acaba4da8590a862dc961b4e3ebfe7a62aaa4b2041bc4af18f957241c7e3d3fcf6ac5453396555ad6269beeb7b45124ed224636067e4b2f18527b9c83cfa53ec2fdc3a075ca39f9bb000f159d292d1db42a189c92703f8cf53f80c55af2a6f236ed65391c95ade164bd4ea8c1b8dd90dcd8ce2f2466ba9a3f9c8f91f04019c7a75a64d12cc93bbb6e45c60a1ddb78fc79cd5cd76511eaa00277f94ae76fe5fd2aa413ac96b12a22f99929827ae4e4d66dd99cce693b2284a914f217328772b8184dbb881edc67daa18f6a29652b1b6de32370039e801e6b616ce27392acf2821432819c9ec0934c92cedc363e612938da547d39fc4d1cd72795f4320bbe18166539da1c0fc3b8fa734f557557dae324e559a4c80076181f5ad78f4e91a465124a922e7e56cedf6cfb548ba6dac7133380c23f99c13d4ff9cf1f5a722fd96a67dbc4f0ce18858dcbe42b8da1b8e54e4719c019fa67157ee219e19649bcd0ece4796b231c7a9cfb01ce3f962a0758a198f90cfe61fba41014f1d79ff3c54aeeb0c3f665884a044036d6e80e4901bb03f5f5e6b377b19383e8320b88cf96ae5c6ee51874233d79e3181f85366bc32861346c4e02a647dee410723b0eb56d9b4fb9d315044d0ea6195b11963191d486e99faf7a7dbd84770cf25d49828702319cafff005b07150e162e5426d5ae52b692e7c979f74c2795d046bbbe504900e467818f4152c9732c8d27fa4bb2ba818cf63f4c7a1eb8e952dd5a44824d8195114606ee7079c1cf41d3f2acf96e1eea668ec6377da4af9efc003d383c9c1fa538d34b56385077d58e90e4adbda32aa81ba493030dc7439eff00855592e2599191258f715fbd1e39ea4f031cd59fb35a4102b895c4c3e6322b839cf73dbf2a6cb3b4a1e4281a4f9773be09e071c0007a73dfdea9bbec2ad177bf633191eddd6e642771046ec671cf38fc38a004854a4642308cc887b363191fa8a96795f219630e8a338c9f97b741c1e79a6a96e06d12ccac4286ea73ee3f1fcea96a73c9b6b42bb5cc9b3619e180062487023ea30072339e477aa8d717233198f2c41ee4295ef92702ac930db9548e60b100ca724b163f46ec3d3d314d5316ee514b2807f7aec471e9d40ebdc73576428c48236913acae327e6f98e3f4e95624472fcab20700a48e321b81d7bf5ee3a71d6a37f2514c8d24a64fbaaa71c8f507f9535dd591711831f1b59bb773f5a56ea38c5ea5b8ee6e205f990c32280a4c7c071ea3f1fe74c6b9565dc646279c9241da3e9dfeb55d4b473aa995c2b618aaa120e7a673dbe955259172eb2804a1e5631b883db38e055289a46f02d5c5ccb1b46c04322118df905873900fa7d39fad33ed4561390ae0738cf51dfdea8e47d9dd33210cf9e99f51fe34d48e662bb13200c6ec77abe5173cafa178deed43b5537e739273b47b7a55199de5d80cd261410aa7042e7ae3d33814b3b66355fdd819cee519663d31f415556722362fd003923f9d52562a555a0318dfb896257f876fde19efcf3488083b4cf210bf743f38c7bfe34c329e579c83c923f952011a14dcd873fc2393cd5584e7d4e983c61c2ec27a024f43c0391fe4d3cdcb23e23030c72595718c7209fcea8282771500a0f9b00e08ee3f3a563cc6ecc57737cd9200c13d2b0b1c65d0ac1a549242cebc13bf8c8ebf5c54a7016158d43978999873d3777ed9e07155ed4c26e645de1e39039ca839ce3bd2897cf9c00a151906d45e47d0526b529377b17614430ae55541c3703702bebdb1530b931b0c363e5ca055c67d07d79a856452fb903f396cb2e471d7f91fc2833f99b22c8da59464e0ee3f5fa54b571b6d1625492ec46cc31bb9ce3b639fc707a559b42ad0dccaf70725770d99386c8c0038ce41e9d3be6ab5a97f26e9a2902ec8861f1c2024fcc7db193cfb5557d50c974d08b77f2d8952d8e49c1e48ec38edeb52a2dec69877ef5d9b71492cad100f1f1cb6d52a8587520f700e79e0d5db66554dd24d2a09188c1620e075fc32a78e95cfc3750c6922eef9430242b0233d78fa81cd396ea3f2e15899822ca33f38230467d72723d29f2dcf41d447569f6796cf74b3ca3232c24981c6727ab1fe5dea1758a384794f1ae5771546277f4fbc4f5fa73f4ac4fb612a85a0caf2abb890063f33c7ae0d579a46cabec740c32a0fccc40fafae7dbad1ca394d281af752c171322ac65e24386dbd779e4963803ae4003a556291cd30b68177c8ff002825b2723fbbd02af6e9eb59cb737174e420254823e794053edc75eb8a9bedf7104b88608e376e36ab600c0f53c9cd5727612926b52dfd926f35e39a37dfb46e55001661c0fa7af27b76a6c8b2c1696eef38f2dfe58a1c64f030496e9c91d07a67bd569b50ba0838f91c92fb140538edefff00d7ad8f0fc2751857ce9238842c49df1e405ebc7be4fe342f7371c649be5455f0da5b5c5fc9f68601547cadfc25b19c7d7afe5576f74bbb10c9797106126706143cb63b64638f5aeaa3b0b4b4821b2b5b70b1ef1208d8e598839cb1faf6ad1bdb577b7fdf138c7414aa4d3d8d57bdbf53ce618258a320c67206d018e4631f953dad6411c8d36d30e40054118fafa575f69a4a4e492aabb5b0063f86b586936f2c12c0ca11594af03afd4562e4dec254a103835b743a6adfc79df04a496438201e3f9524b72515e56919c2fcc0b0ee7b9ad3d3bc2d77e75c689712b4768e0c8f2601f3470028f63c93f4aceb5d36eb4e9516e6668a78251b5146edca39073f4c5755b9526529e8e2656b57705c5f2489b9919232a40e48c719f7ea7f1c56534ab1c8ed8046723279e0f157f59885e6ab717515b858f7154545fe1c9c71f9fe75892b8619ddb971853d8e3d3fc6b296a7995a169366c4172cd2862c37392c188c053db9efd2ac477e9e7c722ba9062f9f2327d73cd73f04ac10e146464f2738c9e98f7ab02e56441132a81d49078fafad351223292674726a91b6c9711b2cd938c6179391fd4546f76cfb9cf96aac36ed700018f9b9c7f9358286300382700e157ae703e9d6acc3e5c663daa4afa83d3dbe9458d7da37a265bf3e691c7eed5d70788c75fc7f01fe1566045320dcac7386f93a95f4c9ff003e959cb79e59630ccd1a000b2b9c927183c63920d4b677024472e7cb6518dea393ee7df9a4d1baa89686d37eec111ab6e3870376ec7b9c8febe94c61746d1d65ba0d8f9826e2a57d3a76edf954515cc9118f6128c73c9c383f9f3d2af5b4ca0280ede580001237f23e9cd4b35e752d8c992e66b9b40b6d64aa107ccccfc484918cfaf03fce2951ede3b658e3b3db1818f2e3240033938038c7f3cd6c4ccd3299a07401b2662063dcf19fd78ef59b0594d7f27eee3678ce03145c85c8efc67bfe3f4a189a95ee8ac5e32c154489b891f3919627d7d3b0e3a5367b989b21d151f059420c019007a7b138fc4d5cbed3cd93c71bc26369324195183918edcfff005aa02da7daaa0974f9ae58a2bf1d0923a1c9c0e878edc508c67195ec654f233b2ac4c183a8dc55492c00e0e07f9e7af154da691146e468c30036f5c8ef8ee7b56d4d02b4667b69372292088c6d6419feee7231d38c8ef9ed58bb58c8cf92aaca54ca40e091c7eb8ab48c651712acd26f2a1d1a4046795e83fad3269d4aa930ab15c1501738f4fa1156163531ae59c7392a89c633d391f5a46b7706364b5c230e30c031e79cfa735a028b654c6f1f204c372630793f876cf269c6460c5818548fba11738fa63a54be54c72bb99114e301781edffeba8658d4ca8ae98cf50a002c7a6700e3ad52b1a455b71d15ce13e485e43dd5b183f51dff003a88c8c4e111631d0aeece3db1f4a1ad632db9023b676ab3395c9fa014f668f6059248d997805413f9138cd329c6fa15de499572b0c257be54b003df9a492ea674408222071f2a8551f8629d2047f99849bc3139c8000fa7f853599020caafcb9192db47e03a7e34c5c8910bcd26080e848fbdb1f807ebd2a28e26c3219be52b9214f07d3ebf4ab41a3620f9285b18c8973953f8544f2292d1ac4548e3818fd4d526274ee35a12a4926321ba80391ffd7a85eda325641165863924124f3d054a15d8f19231d9ba52f97e5aee20e7b06c0cd3b93cab62fc52a3ccec1c28894b280b8c9dc318f4a50639ae63914152400d9c0c9c1f4efebdb3f5aaa271e5941901b1bb0a39c1e39fc4d4d1e0e48618f43c7e79f4acec725d5ee4a4aabfda7181e661971c631edf8d3564196609b4e7071c151d302a74936490f9517cf1e4b0cee3907f2a8cc770dbf119791fe6738e41f5007615362d53bea5d552a246452d18c9393c7a67dfeb4a1cca11672830772edc63a640271d33ce7d7af1d338cb2a1224ce7772fd7071dbdea281bce75dd9db8c807b1f7fd6972e826acf5350ef4b53e506c9392003e9c7b773d6a3f2e5544df21039279041f7e41e83a7ff005ea21732c6c88ac38e0153d0fafd698970b2ee124aebb0649c6e07f0f5a49590a0ec4af7113c326d8c44e49076ff0008c6063df048a8f72daba45c32ee0370e467f9d4458b31f99724f6fe5fcaa6688cfb010ca0b0f9c7407d7daaac3e66588267322b4a9f239ef9e84e3f5c53d2ea6251c80fb4940a57233f4aadb4c8e224dd0b2b6046e776707d78c77e2a6db1c2a048c172cc40e84118e307fcf0696ec6e6d325909dcac58175f9b112004639e3d31cd44f395ddb4989411b9892cc47a60f7cd4a86316b3ce629012cb1a966e3bb30fc80e05574656528a8195886c839c633ebeff00caaae69ed5db41e662542bccfb8767c903ea01e6ba2f085dbbddccaee4c719deaaca0658fb0ec05739e52c9766492405948c27438c0c64e38ad9d26daf65b3d4e7d282a5ea22984f001721b18cf00f4ebdeb29a4d58ba327ed133d5b4ab7937bdd4ea7007ca0f19ad6bc8ccb6f9c6de3ee839e6bc5bc3fa96b51ead6a91ff6a348670d70970f21053a0f918610e4f63cf5e82bdbb7a346371ef92477a99c1463ca7726efcccad6b6252153c03d4d1248b1cbb7b63bd1a8dfdb5858c9713dc2431c64162cc0003be4d7369e2cd22ff514b5f3fc8b9902ba4720da59586548fa8e7152a1eedd149de5a9b57abe6c1e62f12c5cc6c3afb83ec6b9bbcb96bc57861dd279872dbb811f1ce7bfe15ac2ff007893cbf9d54952456540f9d465b754397ed8e99aca52bea6b18d8b234a8adb4f8e35418ea49fe226b97d57c229764bd91113af2500c03f4aedb5093322c11e49418c9f5a85ed1a1854b36413f31cd2552cc52a6a71b33c96eb4ab9b51e5ccb85c90c4038e9f4e6a932e1cb6180200c7424ff00915eb92a457a861f2d5941ea57a7b5735a8787225632c2814eece31d2b68d65b3392584bad0e246e8d154bb1c1dcc31c75fe7c54f14372c8a7ca5da48209ea07f8568bd9c5049f32a9209652a7bfd7b5325255d97cb0b8272509ad5493d89586e4579155b7249f78960b85cf4c8a91432a06565ced571c7539c1fc452caaae0904e580ebd47d3dba510a2248a702403209cf193e87eb8a4d1cb562d4d58b86591e08981076c807ceb93b40cf5fc3ad598d9ccd2bac6b244a76280e7f8782cc7d339e2aacc119a381481c0242af6e723f1e2a48591e00a1518372bce723b63fc7a5485297bd72edd4ace1982c8f3a90ff0032e7773efc1fa7f2aad9319f33cb31386c0611ef0c727d338fe82a4f9c491a895141c80431c71f875fc2a7c3c8c5a29617da7e4f9b6ee03ae7d3e9f5a5a753aa35bb8c5852f8798ed2321c2ef4c01dcf603ffd54d6b2b445957ed6b0ec1f307947cbef8079e29ecf6f319dda112a390ce0c7f203ea0638239e98f7ed4e69d55239513cb8957f75108875ce33b4039e7a73cfe141a2a89abb2b3dbdbc255edee23c9e7cd68c924f6cfcc3b7a01f5a8e654b8901957cb0724b041d7b900f5e3d2b4e3174f3472468922b32afda060b019190476ef54a753733319606605010590b12771fc73cfa0a1b31a8f54e28a3790c11c24b96c467639f9b0e7271c03d71838aab0345b1f6ec4763fc710ebd33d3f9e6ba286dda3b50ab7bb9b63a4922444b0cff00160f1e9c7f2acdbcff008f9323ab0620828ff79fb6076edebeb56a572a0e56d4cb9c086505eff68281d7f7654283c81c0e0f3e9546779d50379fb57aef046d61f5c7e35b12c72b10cd2c7187e76ca83730e99e3df1d38e98cd02de2b52f348ec1a5619573c31e3a8039e9edc7bd5737614938ab98de43c93860cc43afdc272083cf43f86315135b322a81164b670776781d78ed8c8fceadf98257264f2dcb0e248d719c1f423dbb558b81241210db6427d3a1618079c8e7fce69a958849b331620626054823041241dabc8edd0e7bd3ded5e33ba26077738e323df07bd5c904b22ee8b7ab1cb606cc607ae7fcf4a0dadc488b98d958723cac1cfa74e9f9d34ca8a6999f2c46165f3604e47dc6507cc3d7a638e3071dbf1a1f25482c030390140c63db22af4d1b44c56e7ee0c1077e777bede7fc8aa724911242431a100005dc6e3f874fca9dd14c872c5b73c388c1c8618dc47fb3ef556511cd2164cf2d9c1eabf539e6b496da6903b4923e554b9f9140503f1e9fad40f02cc3739f3230436f8df39edfa74aa4cce5b19a1b6c64907ff00af56d9cfd9e2de48607fdec66a38c08e4390cb83ce4f538ebfad3d50bec2e0964404907033d4d3b1c36bbd096da6657c261f1951d303f0f4ab02668626957765860ed18c8a8233b5e3190c029c29248c10739c62ad44237389cbb285f900c28271efd45234836b62b99639707e546c6012dfcf8a7c0aa2524afa01938e09f5fc6a1f904a4065466e0e0124f1528f2c00b26f1918191ce0d160bb6eec49b891a10e83a16c38ce403efd79a9374651d8004b1e76e39c77c7a7f5a8d5c63e760485c06033818aae254591b7173213d4631f952b035644e59500e0f4ce47a52c7384925dca0118c1e9fa7bd55de1b68e5940e481938f4ab52489b14470aa12808e496208c7cc4f523f0147433e84b1f972473838fb420dca09c2b0e4303f98ab1767cfb58e46fbcbf2bb0c6391dbf1079fa553b70a378997911b36338c8fff00554a650ee84c4a0b464103238cf4c67b54bd076b8232b5a3925b218329ebdb157a3b74668e60a42b1f981206de38ce7b1238aaa90830caf8c0257a0e173d7032339feb5244ee878949463b4861c63e99a87a5ec2b34875d379051a40d975e32b8c630335d67832656b9b8b562a1668b3c75dcbd4febfa5731889da25932cd6f95ddc9c64f7fc09eb5ada2f949e24d3592ea165f3d3e58998b15271e9df3eb4593d0da9cb95a67a37ce218e5949558d72cd8e7356209da672124971dc6ee86b335dd5ac49302de4202b6e7f9c75f4fc3a56c68491be9a92e546fe71bb922874da3d1551333356f07c3e2590bea12ccaa806363118c771dab3ad3c21a3e99a933d946f757b92c5e6236c44f5620000b1f5eb5dedb324b030dc0fca4123fad5394dad921b78106e619c0fa7249a395a8e8c39d396a504d3a3b7b3545ea79e3a9a86ded92cef5e427e758f73b1ebb8ff00faeb520002191f9503a91d6b36c95ef1e69641b448f923dab9eaa49236a6db6db218b2599d87de39e7b0a6dc891d30090a7803af15abf648848c704a81d3b540f179f36ce8a3a91dab95dcdd34ca0765bc1b01e4f27fc2b0efe679dfc8690ac63a81dfeb5a5acce90af9510e4fca3d6b2ecaca4966031f31c678e87d284d97656b99c6089632ab18e3938ef5977065fb60f2e1ced6c80c7a1af4afecc860831b06fc64b62b89d547fc4cdd42a8c7000c63f1ade2da327252d0cb92d46cdccccaa71d3903d87ffaaa231c676013c6811b23318273dc9f5e3d6b5da5458b96dcdb867d8564de4914c595630c738f615aa93319518100c8b9dc235618e485da07538033f9d695d451dba59c3868ff00d1a02eaadcac8c9b8e33ec47078f6ae798c968fe642e464f4ea3ff00d552cfaac979feb4b094019607ef103193f8003f0155b9c55284a3f09af02832c78566c062a8e39cfa7d79fd69a5dd57cdd88bfde3229040e4f438e39f5f6aa76d2662562ec700e5430c67b1cd5b1a830689222ae4801cb8e847527a820f5ef4ec651ebcc4f6e1eeae0a246b2b151f29621b39c018f7f5e4723380334e93742e49887964ed8d84ad8620e0fb30c9038cd548f51b949e416960b712ba18c4d207c7fb981cf6e49c7a6319a7ff0069cca84cd32cecc0178f6edf2d813c20e830076e2959a378be55a96414381999546188494f257fddc11f4c53ddb9dec493c72bc6e3c741cf2075fe75456fc456ec42a1438292280483d720f271f953ff00b4a428ed2b3484702373bf3d8751fafbf7a1a3584e325aa2d9baba9832e2dc3118669806c8ea41c6d3cf6fd6a588cb76cdf678fc848d400a63462a481e8704fd38e3f3a306a0be4a15890951872d8dc1b3cf41d3d3af4fce69a6f35033790f1f2703e5624751c8f7e9415cf14c918cb76ed6d209662a092565c13ec4280071f5a8a1862898be52dfe43f2492364af1923aee3ee39e7e953bb79c81dc8b503e56793239e9c60e7be7eb50da8d2e2de92dc48eee14e62b764c83e849e475e460727ad1706e321b2c313481a48a595b93bd4b600ea00e318f6a8bc974665876a9e3e4caa11c6724b2fa1abd34a896e0081fcaf2f3fbcdcbb724e067ae718ed54cdc5818fcc49269254f97091ee500f23e76c64e0f6cfe3426178b285e2456ce7ce9962775192a11b3839e4e063bfe5daa096c6340408616655c990c20127b963d09f7f6ad33224e07d924b86db9dc8003b0e33e87d7b0f5aae82f13747184c381b9b6f320c0e871c8aab8959944c704f929b77024c872c4139e327b63351358611769691b729558c2e57be7731000f5c0aba64b94722299a42cbf3290b84fcb9fff0059a6b2aac61ae564b829c83c654f60011fe73c74aa4c5c9a6a557952d90332c11bb02b09c172d9e589c9c10471c75c8accb9dd24a2411ec4070a1d80041f51ebe98ad792e6ddc0758679a4fba17e52fd7a0c1c2fe64fe5549e64b890b4d6d6f12842179dbb73f5cfb75c67d6a9194a317a197922411499673cf6e3fc8c53cf9692333a091d41ca96e01c6727d7e9513aee36e7a7ca0673c11dbf1c1a5764049909651f746385f5fceb5f23823a31239844cd9008c9c9cf18f4febf855a0181daedb1f6ee54e3919ed4d7756508d82a0f2a0e09e3da9228cccfe50ff59db61c64fb64fbd052b0e85bf7c0e172dc1c9c7d3a7e3f9d4d20112abb05243fccace0e57ae7d3ff00d751e238a54dd927abe1b8f61f8fb53d5b74e080559788dc27461c83f5fd28295925709160f33f7477c61be4653c74e323f0ace8ede795cb4492160436d0a7f1ad496f198012ca5c905b2eb8e703af6cf18e2982e4141e5b104a8dd93c123bff002e94852772116ae0ca9231888c798c61661d3a67b7ff005b15348a82dadca857c6edc8109f30e46011e9c9fcbde8171238f2a36604f24640ce79393e9934f53b3686003e738639039ebd2902b3238a1446382fba4608001f393ce7e5c633c54ad3c7f7844383bb1ea3a01f4ff1a9229545d8751e63a30f9dd727773c819e3eb441670b42fc0e570002338cf603a77e691325d821b90ce4aee59015eadc7a0fa13fe14b1c9e6c484ba85dcd96c127bfa9e3b76aabe5050cc9cab11c3633fa7d2a69143ef50f860db463bfaff5a868877ea493048d73bb62372e17927eb9ed5afe1a82e2eaeaf7ec223fb6c36333db9278f38a80bc63b6e38fc2b015e395b73380c01e59338239c7e55d2787b52fec5d525bb788caab1012146070188c7a0c823a55c5599a415b7389827b38ad96096def63ba40c008994f238452add00e4120e735eaff0009ee6ea6d375433197ecc1923877f0378cef03e9f2e6b8bd4f4fb6b8d51af3ec17d0b4ec6559e143e4f24139f2c92add7918e49c83c57a1681e24d2f4dd3a1b5ba7860941f9521dd8e7a638aebba68e85a6a8eaedee16ce47123150cc060fa93515bc96b77ac5c441c99222de61c70064607b9eb5c16b5e2a8aef52436e6492085b78da0af98f8206e3d94707d491e99cf63e06824834692f255776b86de646ea48ea4fe35cf3e54ac689f33b9d05c70822c6d078f7a62c51461625c051cd2870e771e49e714f5218938c93fa5799397333ba11e588c9436ddb18eb55ae145a5b16cf27daaf2603163ff00eaacbd4333c810723ae2b396c69039f36ef348d34a09e7e5f6ad5d0adbf7ad291c0c81f5a64a9e54640279ea2b534c8c476833fc4734538fbc55597ba2dd902376ce07ae7a579c5eb8333c981924f27b576bafde2aafd9e3382ff788ed5c3dc02f2b007201e87ad6ae4ae4c23a1993bb32a8c9c8e6a8c8f21e83fa62b66484633d01a856d86d04fd4d5458491cfc825c743ed9a83780794dcdf4adcb980104fa567f94158b7cbed9ad93460ca42472df771f4156145c3284d8e50f3d7fa55c85519c00010074ef565cec8b9e58f5c76aa4ccda466912a9fbcea7a75a4646247ced9ed57d1033618e6a392307057838e86aac4bb147f79028c31001cf07f5a9e1d4e740cdbd1cbf077a8ce3ebd6a3963233c1e476aaaea55f201008ea452b0591bf0ea16d70f1c73462300807737ca78e3bf157f6225b2b5bc2c51cfeec15de091c71939c8cf504f6f6ae595c3a8fe13e83bd59b7bc78f742e15e3231f38c81fe149c45249ea6f8fb2470a930912b9d81903237b8ebeb534335c4d1ba418b7c464448382e47dee472bd493556da36684baa94c92c8d0485f8ff006811d7b77edeb8a98b2c7e5324b1feeb2a0ab901370e78fa13d6b36ac118b5aad089638e194adc289a400348ae83a76c9623bf393ed567cb585e776913cbc6f60db5957fce7a671e95133158b952f0c4bbb28ca4018e7a024004e7af7aaf25ec36e893c5049215c1760833c8f43dc024f4eb42bb60ed7d4b02678ed1550492862436117d381c67a0ce31ea7f0a332ced18791a08e30182b86ce31ea08ce7af534bfdb6de598d6de49e305d9a33863b8fe3d476c71d6aaf9923dda88439f3a209246d6e4b480139de4120e78fc6a947b8f99a5a13ef98481122deca41657e43123a7200ee3a1fceaa49148632eef0a2e794039273d17a8ff22a1f3648d56378a6d89f294946dda076c63ad3da7f3826205316dc29f2f0c0febfcaad20e772455b85223d8ac39ef8c1cfb91d2920b7375749e6cbe5329f9e7d87818fbd8eb9edefed4b73692dbcbbd6768f1939da32073c0cf6fad31cc4b1e64b894b633bfc9cb11c743cf154918283e6bb3242a24ed23ca0177e020c96273d4f6ed5644313a831f99b98940703838e87d68f355e6549217de0eddcdf37d33f8f7a9add445b982eecc870fcf07078fc31d7e9dab56ce5b762bbc20f992484ab82140cf14ff3711a34ae0ee62010063b75fc7f9537ca8c61117257b7423ebfad32e22f34a9594f1c81b3803e99f6a92756589aea4b842a58fda235f98e387c1393d3248efeb514370a90ed0763264a74239ff1a8047292417cb02a4163b768ea4ff2a9a5b6575468c30dce77606428c9c74f607f9535e617b08551880f21c718c761da9ed000e046cc37023229ac18b9603e63c0c9edfd2a58d63758f7e5a404e4afcc1b91c1f4c629390e3a8c400c8db91b2b8200ed8f51fad598a1dad2b128cb8c2904a839c0e83f1a72ac493075c0556dc763640fad3e3982c2e138571c60f3fe4714ae34ecf517ec31aa248858b3601e085fe5f9d11b88e391da55e080a0719c77f5cf1dff00fad50bb096028642a4646f66dd8cf1c7e55564917f785090ac37367d474a4ee36b5ba2db0872d999b84e7e4e87af73d33dc536311965506777ea3a0073d73fe7d6aab60176c6339600ff009f4ab302f9f6ecf2b84208c91ee7ff00d54589b6a1046e619bca7c2a6183336063dbb9cf356a0765827c1de59d509ee40191fa90703d39aaed344bfe911abb36f2cd1ba9270476dbdfdbdea770e90a5b0442e1986d53c1e401d3d78fa90686b62a51b0eb3bc9edad256b7b87899a4192add78e73efc0a634d33317b999cfce36ee6ce78ea7f9d451c4cac2de4631acac115c9191d76b0cfa1f5f7a7bc57593110c1e325582b7f1038fd7fa8a352799ecc25bc9638fcb8e47550a015078dc7f9f5af40f01f88e59f1a1dc4a02ec2d17cdfa7bf5af3b58cdca3cbbc33db80ecbbc6193a1fc8ff3a92369a09711c863955b8707a0cf5fcb3f5a1f66546a35b9f42c509f2f0572d8e31c03e952a6238c92304761dab8af02f8c86b11359de362e220047dcc89d9bd8fa8ae8752bf5b7dd9c00c703d2b96ac54353d3a551d4b1335dee0d8c6735091f36719cd410b0768d8a8c1ab726093c718c60d732573af6d0a371c9272476fa5682ca12d14f5f979c5509780d9031fca9b2cc440108c1ab8e84cf5b187a9ce5a52cc79cf02b20733606339e6ae6a07321eb8cf3c555527cc076f153bb34e656229547dd230339a82560a08c28ab13b2eeddd307bd55997703c1e9d715a4519c995a7e62c8073d338ac7b8215f68c63d2b5267653b4f7fc8d64cbf35ce4918ea335bc4c6459b7c094039db8ef48ec5ee081c85fc6991b32c9c8041a3044d83d2b44ae66cb518e339e33c669e10331c6463bfb535324727f5ed5614291d4fb0aa208c5b824f1f80e99a86e2d91970540cf53e957b6ae33c631d476a6b01206e3ff00af4c0e62ea09217f9092a4f4a6a4cecdf3673eb8eb5b7322f9a17b13532e9c8ebb48fa9a1b2ba95f4cbad8e15c061d83738ab134cd04c448e587552cb92463b30c739f5cd64dd5a4b692878c9d80f635ab14af7b62e10959947ca41efe87eb5364c2f61eaf725f72e76aa056640064f40c58febdaaadcc48ce2631a24aaa30165c0c8fe10738f5e71c8aab1ea32c4cab2390d93f32a8e327ae0fd2ac2ddcbccc9046e9b88f30a824f623a8ef4acd13cd17a0e6937e0c92c6a323219820e98e581c719fa537385558645dca70b2a8e9f4607ebc77a62cb6acacd3428b9393c1007ae47ff5e98a626fdd0994e586c551b01f51cf07a0efda9a571a649be58c95311954f50dc1fc07f5149710a40e5229dd2e983623760df2f4c807af38eb9f6c531f7b0c3b6c0f8c45f74fb631d78aa93318c3386e5860072463fce053481b1cd74cad99159ddf09feafe7cfbe79c73df1d6a19226c338d92374322b051f967afe94f7ba9bcc7fdf0518da199b3c0edeb55d4c9713ab48c1dc63e78f82463b93c0fad68912d91cfb99d4aee552a01f4fa1fad58844824484cb222952cc3b67e9dcf22a489c19432121376e0700719e69d36e49a7565538ca80b91c0e9d683815ec36684c8862f359bcc6e99038f4f614b2d80dab0a624766c29507951d4f1ee4727d280d9870491c64e3ae29b1bbce192431884e000c7231f8d24ca5e60b602356492451230f949e491db383d7dbeb4ab6de59cb4b840ea4c47382a14f1c7152dc16131474da5173b5876febd3f9d4186f997cb25883923f1ffebfe549c89bab68324b7daaae24664c0c71c9edd7d7e9eb49b625519e1472548e9ed4b10011149249f43c6ea7f991f96d17920bb6406c723eb4876b15b996497693b76f5e9cfd2a63ba283ca60c4f2727b1ff003814c8810d24a77001b95c75e78fd69ec06ccb12bd41200f9cfe7f8d325a7b88600b1baf1c614103a7f934e368f2cb952f28e0b71c607f9fd69ab71ba7f9983247f300bc1e7deac3cbb7ce678954f97b719c904818fa763dbad05e886c76a8d691cc017126432b00460741fad59588245b76ed512a1c0c00c339fe42a9472116f141e73840a3207e5802aec65aea07d8019e41df8071cf1f4cfe14a4ed1093561b6b1c6b3fda5198312554027b83f9e3f9e3d2a29ace38b4f6250498896250dc904e72719ec0360fae2a4522511ac44b29f917736370f5fc719fa1a959e416d144caee4c8418cfcbc9f4c723803f5a571c6f22bba34912485433ae1642d8f9d7b67dfafe5ef4e07642afc8746063900c6f523ffad8c54cc20cb2ed950ae0b2e0b00491f97d6a7100f2e4ddc46a9bfe61c023d3dbd7e953264ca16467a47fe84b285c3212015ebd7a01f423f5a661e48e7959a2464f94b3e067a70bea7afe59ada8a35db0a9511b2172caf9e70cbfd79fa5663d8c92a7ceb827192ec085c9ef8e73c8e284efb8d534dbd49bc39aab68bab5ade4caca8b27ce41ec319fd09fcabd1b58bd8f553135a389222dbb703c102bccaee1821882246a1d08c9e49edd4678edc56cf812fb025d3645298cc9106ee3807af3d79fc69558734343af0f254e763d3ad4ff00aac1ce05492cc49e45436e36afa700557f3099482d919c5725ac7a0997401246d9eb8c55524c9073c90783d2a589f0e01e3eb51483cb9dd79dafcd5a892d9cedce4ced9aa61f236f1b874ad0d5236597728e41c1358e5b2e73d47eb49c6c52912b9c9c3360d37702a704927a531a5e7e943365b8c74e82a9221b2a5ca81d3b1ef591731b23ee1ebd6b626072431e7b64f1556540e8518807b119ad119c8cd591474c0229e92abb8e79c542f0babed39e7a5244b89b049ce315a4599336227c28ddca9ef53a8e43039c0cfd2a929f2d80fc40abb12b303b4607a7ad55c2c399b9c02327d3d29803799819c1e953a425db3d0f19c1e95716d0743c91ce4734931d8c8bcb29546f453c7cc6a4b3998a8c93f435d55a5b25cc6b1c88a7b06c7355b57d192db64b0a7248c95ad634dcb63395451d19873c0268ce5723e95936e86d6f02a9f95b9ae82105948200c7519f4aa17f0e7e70bce7ad676b3344ee626af1bdbdc6df2d1e1906e5575078ee071efeb559a6579b335ac72391f2c9b40638ec4800feb5d1dea8974edebf790646464ff9c561c33c5bf7b3194e090ae4e0feaa3f5ad2d730943a8c8c4ab2325bc25c30c8500823dbe638c7e34d669671cae10e72b9427fef9c1e2992490b061241210d9dc8db401ce47193dea269d946cf2413c10ad200738ff3da8e51ad07bb045dc98661c101f1fa607a53164dac730a85cf0e4727f0ff00f5d39e148a767022009242950180e3a81d073eb51410abb90f31765fe11f283f4a2c55d0aa53cd2ed1164c862adf2e7d79edde9ed25bef0b19099fe13cb1fc3fad228b867c29468d4e093f281ebd7935123c10b339585431c361465bea4d3b013926160a0ae0e4a93e80fe55624fde8f31fca0ce9bf6a9c63ae7daa9aca248994b1201078ea3dffc47a7d2a61189208090d26f631850464f4e3f5a9d8f3e3269919687928589e36ed009cfa7f9f5ab445bba98cfda8f3d115431ef824f6aafb1fcbc9455c1cb05c9c6338c77a5b78db610ca36ee0719381dc77e4f4a2e93348b2749e38a211c692140b8dacc36f523b0c83d3a1a92d9e23868211e66ee5646392738e3180477f5e79a6ef8b3c46ade99e71ea7038f7c5465422332938639c1e38f4cfe152c7657b8c6019806d8aeabcc4aa011eec3d47f8d0db8b87e31bb903bff0093cd4d0cae3f7737ce84e46d38651ec7fa1cd2bc4de43852e7827254061dfa7638a3a8a5a958fddc6d1c1cb74c6687010e7672063b1c548b955023291f2492c9b881d073838ffebd3080814194bbb1c0cab2e7f03c9e9fd6994a2ec350bc8891c832a092776727de9a54c99e18ee7c0cfe74b973301c8553c0e78fc7eb4e33082de376e4f501f9007bd4b6f6444afb0d8e29259130768041da4fbf18fcb157e22b0ca11db60dc47ca7b1e0e3e9eb5491a4f3cc6a5b7fdd18c8cf7278e9c52b9fb45c04024dbb4aabf703af07f33f8d296a371ba2e00c2744580b81fc2324ae3b640c7e957845148ee1ada40ed28c1321209ee7a71c1aceb196e141d85f9ea558f3ebcfe157cc8c3f7731950a2fcf21738e3dbea7f1a87736842c89e15b72eb1c68c24964eacca4360f1cf51c71c7bd5886d959167b629e582e26b769036ec67ee38feefcd91d47be2ab011c71b476ff328c0e3231c7438ce073c8aba25596d25f2a3dd9c2460b6414e4e739c7181cf51da93bb2adcda1049e4ccb1ca974de4ac04b348b86419c0c8fe2c03d47514eb7b396d65b8c95376be646a25c80bedd4e548e791d8549258a3ccea6258ee428f346c187cfbfe47f1fc6a7747fb42444b332a24588cee72a0051b9890327040249e87349bb6c128a5ad8e7358b993cf813e58f62e42a2e40cf24720647be2a3f0f482cb5bb7b92c3679989093caa9e3f2e7f4ad3f13e9f247340ec901dc0ae25baf34f1eb827f2ac42e446d24c70a148da067e63e9c57441de2435c923d9d640c808619ebd6b39e4315cb1e369e78aa1e1dd47ed9a5c049cfc8082dd71d39aba183c872320f35c535691e8d395d5cb29739913e6e4f6ad0b901d54803a561dc432261e3c9c738f515ad6f309ad1181e08eb5a535726a3b1997c9f2e08cd72f75847cf2a01e95d9dd8578cf19ec6b98d421fbd8ede94e50b31467746634a01ea47f5a707dc7a8e2ab321208047b6692395892acbcd2486e4596391b4827dea33b197076e7da90960464802a458cbf3c0157622e5478449d0fe04540d6a5195883815b0b0e08cf0474a4ba11b44ca71bb06a9225996a4798bb8e413cd6a2c4a0019c0acb9130993d3ad69c077c1192df5207068b8176cc6d7dac4edad5117cf1b0efc64567da2927e5db9fad6c08c9504e0639e2aa2824c96d55ada6208054f4e2b4eee349acbe6e78aaf0b19136842cd57a584c76a37f5c73ed5d945347256699e7e0347a8cb6e0f4f997e99e6a79add645209600ff9cd3f53db1eaf0caa3a9e7e956a68d4c63e6ebd8fe78acea415cd69c9a31de0d91bab63a7045714e446a6227736470cfdc1fc6bb89bf7714adc850339febf4ae05a793cf768aea30493f328271f863d2a62b42a4ec4fe784913ce3c1200c6415f6ebcd02750444ad1920f25db39f4fa553552ae375c1918ae06d42723e87b7d6a545cc8884aa26efe26f9873d87b556c89ddea2c611943333bb283c8ce091fe7d289863e58cb6cef852a07ea4d46e76fdf657c13c7419f5f71488c85fe6784381d73c9f6e47f3a4c64d0b02c410acc581c639638faf34e924769018dc291dda2040fd4106aa9083042b18f19c100640eb9c5219a5701884c6ec0041c0fa1fa52b0f52ddd43f65be9228a40ec8c51f0300e3dc707f4a7895563955490cb8900e854f2b4924b8937793e63124e4924027bed152d916b849239503c7e5b000000ab70463f2e950fb9c764558a4f3463cc70a31b989ce07a7bfe3574c8a1a2d88427ddddb7713ee47f9e9554bbcb1036eeef0a02445d48fc3bf3ce47e5525947821da32f870c5491f3e08058639e39e31cf14a4ae0e3d51652630ef122fc8dc04e413d4647f9e68795590b9c70c3057ae3818c76e94e7062bc966deacabb882e436eedd3d318fcea15776b78e252410e5003839f973fd2b32d26c9ad90197682dbfef2e3927d6ac9b0782795036c9213f3f98496041e781d71f5e6a1fb406b45084ab63076282ad9fef0eff004a90f992c935cc92798f2e64625b9273cf539efe9e954f42d587b1b32434cd217dd82236f2f23dc0e7dfb76a4f26665630c31c30caa0972b838f4c9e474ef54a5f36575e51463e62c0e3dbebc76a9d6678b0d9895e5eeca0923dc74fcfb51d0a6b4d073dbc0aa196612c830dfbb5273f53fe7ad426ce0bddf1118618668bb1503e62a7bffbbd473d455a9b65ce7f7ed1c818f0bf758e01c7180bdfaf1f8d2a58446f8149255dcc0c8cabb4a01c9c1cf03fc695ecf533924f72308913cf23cfb894db98549073d704e3d8671dcd4104af14c4adb490c6148cc84eee08f438cd49758bc82492350ec4e1d3715ebdc6075e99f4c67a1388e3c11288d80c0da4a9f4c74fcbad2bd857d342dc7237ccb232e5b9d80eccf3807a81511b88d5a44112c883f80839ebdb1f53c77c52cd6e218f748bb9986c0b8cf1c0e73c0aad1cad713ec59228f6b05f2c0c91dfb71d292d753583ee6c2318e4479e760701946f65c72380074c1cfe7cd5813c314a73124922863b645055f1d0f3e8790473eb5941a3cee61e6047185de54f3c678239e3b7156ed1d5a43118d17cc25784270bfcff00ad26691926b436a2f13bac31c11db5b63019422f99b4e41c177393ebc0fcb8ab4753ba6750ab1432a49840600403d4ae08e33e878ee39e6ab4d6b6b05bf992461e3d8772a7c98c609c06ea0e4707a73c9c5466e259ccb7f6d64f2ee05653e78fddb02aa091d18b0008eb8e7a1cd4357d51524dee73fe20d42f350ba68eeee437d9c651591502e719002a8e7bf359a80c918b56b811c887960b9f998f048ee38033daba39a04b99c89b73856dab893a93d000401f5e7b74e6816515acf2e2258e4519764396c73c82718e9dbd6b655128d8e76b99ee3bc36f3d842f6f322c6c921645e06e53d4e074e726ba54b962e1800c3a8ac8596e2c2dde2f29da097686271c367e5e79e39e3ebf4ab28ef1e1b008ce09f4ae79be6d59d941dd58de82e165ca9186f7a96d8792922e7080ff003acfb5972463f3f4ad158b7c6cc78dc3057daae8bb31d55a58c39f5a02fe4b664395ea40cf1fe4d2cec93c4cd9c9f506b1ad2f035f4c6581982c85378eb81915ad1c713c6c2363b09c8f6aec704ce45368c1b84292152c33f9542413cf0cc3deb5aeacb73165231db9ef55522da4f241e9c8acdd3b1a29dc810b9e1908cf38ab51a11c3640a91564001c6e1ea3fc296524c7939040ee2a39595cc8a3335c07f2e375039c1aa855a2c0924624f5c52cf71b24e7b0eb558dc2ca086ce6ae9c53dc9a93b6c5c7c340482738ab5a6b092dd70790706a8c64f9386c8c75f5abda2a2f972f43f3e79e2b26acd9aad52376c630a401953f4e0d6d28f900c645645a9c6383d6b5a276046718fad6b4cce659b2f36d666641bd5b9c1ec6a5d4af1fecae5d82e46314e4e390bc639ae73c4d7a515141239fbb5d29f2a39dd9bd4c6d52e3f7cb823e5e7f5ad88e117368a724b83c015c8dddcb39524f2724fd2ba9d32706da3201cec1c54369b292b6c665d432bc3716c301ca32023dc579f451ee40137924038e723f3af52bd858dd432a00773738af30bcb6315d5cc0401b2561f3be49c138f7f4a98a357aab88595119f3d7a7727fcff4a837bfc816295c1eb9200fc8558b68e59d83b05708bb42608cf1d3d3b7f9cd3becd0fca2430a0efc86cfe00d326e51da118970c32dc16ea4ff00b356e25b64c9753b8804724edfcbdbb0e9565a383e671230dbb71bba9fc3038fc6abca92b2390e483dd48dc73cfe3cd03ba09becf1ba96dd33f523cac7e1d7ff00af50b4cb094711955386db236e38e9ce7e94e8e37dc1bf79c73f3e71d3a1f5a7ed85e422e59642723a6464f6ebfa0a571d8b709927dfb26112449b9c81c60f007be7b0f6a92271f6cb64b6f30411cbc07037104e0ee3dc9ce735465b9054448be5c4a7852792718dc7d78fc855ad3988579247c206c6f62480c7dcd43d8e7495ac5bb78c35c3c5200708c03200878ed91c83cf5a94a62732245e50072db9b1cf41938cfbf51ef9e695a678ef88645562abd38233ef9e739eb53c2e6780b309a29011b5836378f7c0ebd8566a424f5b32acb1bef7410176942005464b02c381dbd7f3a60b06b585562632cbe73ee273b41e17ff008af6adb85fcc662832e8abf372aa4e31f8743505a59e2fff0075346bc06e707191c9c7a74e6a23245c1ad8ccb6b2bad91b46c02ed00955c8200eac7a018f53daadc3676d69335d3232800b6e68812c9919240f98f4e3d78adfb64b69ccf0ac8b713b71e4bbed5c019f94f6c9fcbd0e0573d708f1dd4be6338b8dd966518653dba7a7e63144a7d8a9bb2ba1f6f6d135c19a593743048abb2304b4a485c2a8ed9ce49c74cd477bbaef5196728249d98068fa08ce0003f0e9f87d69b791cd322dd141b9a4db18ce318fbcc0fae30a0fae7d2996cf388de09a713041b996e103f943b1dc707afa924d545de2284fddb31fe5cc6da78d83ef8883209142851d3f2e00fc69ce5b0855c08da22464f7238e3d00a432655d6517112b30215b0c01f5e79e6a1d426f3236db0302b16088c60fd140e076ed512d593516a68347e4db589b3625cc46ea6c313b4b7dd8c7381b54124772c7d2912584c6654b911aee2009e11c83c83c640e87b76aab6704292dacb7665b76de46e9064a8da4f6f6e31ef59d8449199a36cf97927ae17af03b1e6a9d9ee44bdd37677b7bcb5d9b6167c6d8dfccf9307a1560707bf06a3b2b8486dc8ba105e1202426488931e072165501bd7ae47d7a563246b6b1bc6a9fb92fbd4ab72e0f39c7af4fc735a91b98514c6cce8e85b0c0631dc291dbd695eda171a91b16e78adee00920b3b8890001fcc8c6179ecf8048e07515359431dcea36715c831c2b23ee21c83e58527391d09c56747763cb2ae372961c9392a06411edeb52da4c96b2a33953988a3a4801fbc3073db033d7d291a464b7ee5fb69a2960b69a45690347e603292ab1f3824e324f1b5768ebd2a5b89c5df976cffe8b69b5046e9f78124e59b1c71b400a3a0cf35993b79d78e2782dc790a576c10e12123e6e17a123ae40f438a4fb4ab3c25573b81014f5041efedd3f5a1e9a952aab62e5bce2dd196e049f692cbc46a1948c772d9c7ddce73d8735720bc8de0d9bc6d7c20708a59475eb839e831e959d05cfdaa2bd9a6748e495238d11d308a986cb1c7f160d32da44b655b8b80eab83b533f3ba81c1c761f5fca938dc88cf4d0e96d02aa8052497cc0b8db3ec5f2c1e47dc3ce7b1e07a532492312ed8818d182fcacfb8afd4f735856da9c735b3aa4170c8c096759362291c73c658e7031dc8ab914914b2b9862548fa0404903f13d79ef52d3b58da849731a624f202ed6e1b83cf4abc9aa14809230c07cc33dbd6b3161678c001714e44f251cc83e4c1ddef4a09a3a26d3dce6b59d5d6c758952d508206f9863e4de79c0fc3f53f5a92c3c6f024856f2d19971f7d3a8ff1ac292fd65b99244b7577724e08c9ce7d075accb8b9959b2dc608da113681efebf8577a6d1e74bad8f4f8f55d3ef62f32da7520f58e41b1c7d41e6aacb756c8f97651ee7a579bdc12b2b79ea0c871d3a0cf4c5085e489d0bb1c0e85c91f4aa72bec2b9e92d7d02282255208e0835467d5d0911a80edf5e2b8485599b6630b81c76073c7e26b4acb292e369eb83c534937a87369a172faecdbdd98e542add7d883d0e6a7b0d62ce29552e2200138cfa7bd49ab5abcba67da963767857e618e4aff00f5ab9372d34919456c37401720fa807f0a528a8bd07cdcc8ee756689a749edd81590007152e8ccfe6ba01c70715c659cce1a32253e4961c76c1e3ffaf5d669f218f504ea370c5615b7ba3a28bf76cceba12a304673f4ad288336081fa565c2cce3b8adeb701e1ca95f4e9d69d3413d040ec3b741d8d715e289d6390c929619e17dcfb0aeca68c658f248fd2bcd7c51379bab040851633ceecf27f3ae85aa39fa95d1fcf7dd8c0c74f6aea342955a03bb191dfdab9489944649e9eb5d3596db730a29e0a02def50d58a4db6685f86dd1e318ec71c0af38d7ff71aedd64150ce09c2647207f5fad7a65d4a7ca19518f4f6af3ff1643ff1378a5cae1e1fc7827dbd08a4b7346f43290bc86075894b4473e63721bd7bf048ed8a9d618910813b31033851803f974aab6ecd1b0063dd098ca0c363af439f6ff39ab0c47d99c6013820e7b8ef53395999a8b95ec3e1b5121792155dc5b9c3866efce79f5fd6a68a28ede1742c771dbf788c71db9ac884a8977c90a6cea197ebc800ff0017f2abf1cab231765cc64672c48653d001d79ce3dbbd53d486decc748d034813e451bb27cd90851f5ea7f0c0a69d88ecd0c76fb58844c272c719cf3dbb53b36f101f680cd9181100ad9f6e7af6ed5202c5374502c6061944984cf18ce4d49ac548cc58d5e405436d1d5987415af69e53dace8ad83160e0023246dcf19e7af5a8a18246d8a577283d88c7e55720f32dd80784812a14cb29c743ec3daa66ee6693b09868eded833829800165c18ce7919f4e3a530dccd12302b85e33b0e770ff00f5e2a579448218e40906198600d8b9241381fa8ab16f6db818e403606e0b81bbb6703dff00a567d476d4559bcdd3ae2f18c87e4551b4720fcfce3d718ab6b11bcbb5713a22cb82493c64138fa9ce0e3bd469670fd9331485a2330520b8c038c9ce38e98fce9901925ba124932c8ef2ec1900141fece3e99fc4d4356611b264928b5b5502791e423324936d0ae4e719c75e3d4f3d2b40ead6ba9dac2d7f6e5edc2e527461bd87d467247bf1cd655a5bbef95e6b756725b781c8d9df767be3ad4e645446b4762654201980ff005aa7be07d7b7ad4cac852a96b91de69d12ff00a9b84921f2768907cbf20e76907a1e7923afe42ab33483c8caa1d9831a962401cf273924e0753eb51c88d9b76690226e3c1e4e738247e357e32f98cc5c285648d146580c77efd07e86a9361169ab956f115a08e4899b8ca3a91f748c7e7c7f23558bac91dd96dd911865c372c7b024631da926b848a43962d1b72cbdb8e9cfaf26aaa5c7973942301b76403f294238fc724fe94dc598ce7d4d249eda7f26d7789628c1c49ca67839e7d01038f6f4acd00af9abbb255707dffce2ad0babe11428d76aca9b5096519c8392781d942e3dea072ac7ca11c45e4f9bccdb827a7707ea2a9a2dae6444b24a6d0bca77a82195d7b027047e07353ade48d6ef13b30091bc8c47f1edc03c7a8e9f9532d8c5242c896ae63f9b0fb8f507b0ef9e3ae3a1a86500a29f2f2c030dbebea3f4fd686975337a3258675551287cf1f3a91c1e7f4c13da9669b1247e5ab8c8dbd73939fd33e95437989d846a525da55941193fafb0ab6f0ffa3cd7a180588c71b807046e070d8f7208a7ca5a8cafa1a2d3b4d0c9379b283022c598d41122b12a5093d4000e7be00ef55d276b795b62a92eacab9191fddcfb104d410c5113044b6f246d6e1ae15b3f798f552bd7a67f3348ecdf66214173900e41eed86a99226a365d8ae3cb75958020128f9e70187ff5cd3a1312abef21db2ac4fdd6000e14fafae0e7ae6b3817855d4b92c70067f888fd0f5ab5b165da84ed4da39ea4b63926848a85f62f492cd31501d52dc7cc3cb7e71ebffd7fe9576ce5ccce0138121038e31daa85b4d0ed8adda395fcb05949936293d71d091fd715a11cc924a0c51246abd7692c7f127fc38cd4b56d8eaa71517a1d144498c15c1238aa9abb98edb6eec70771278e066acda8fddae3bf3593e259716172fbb69116d5fab1c0aba69391751da279fb5b322e2e6ee18a553f32386ce7af040c7bd48c56578e496749d8606f5c861cf7e067f9fbd398830450360c6a0ba873caf5c01df1edd39a8ad91813201f28c63f3ada52d0e56c4658de42ae406538276e7a13c531ad238b61173bb8c8fdd32e73fe7a1ab8516494cb98c6d19c76cf5feb55e7b7479bcc2599b183f5e39c7f9c528cb52168319a08f6133ed8a33b9cf979dd8fff0051aebec95249a5808459eddb64813183900861ec4115c89b6cfcbce3a71f4ad782fd96e6d75521893fe8b3fca01f97072c070090c0e47f7477ab4d171b23bcb28b747b5ce474c7ad71575610d86a52d9832052c76aa75e79e3d78cfe55d709a588a7960156e73543c4f62c6d63d4d610e61cab90b9c03d0fe7dfb75ad6d78dc256479fbee32bb093209c2e474f4e9ed5d869a5a7b382e01fbc467d88ff00ebd73ce16e5013e4dbca72125f2f025c70413d9bfc73deb7fc2cf8b69ada41b4893207a67afeb9ae6abb1ad0f88eced0f0bbb1bbf9d6e405046320015cf4059234e856b5d33e5e47a7426a69b35a885d46ed60b79240a1b6a938271faf6af32bfbafed2b83285755e405690b62ba2f13ea6f656ac11b648c7e520e3ebf5eb583a5c424b5dcdb7e639c818aebf2398aee92adbe5541c0e81b935bb6d31c2cac0af6c37a51696d1676b2f538fc68d764fb2e972491a8dca005c53a91b44707766df99bedf71e4019c74cd727e260a6ce3baf2f7b46761e3b31c67fa7e35b7a75eadc438dc30147344d1821a1080c2eb8623a8f420d735f536694958f3d8d4b65921455c0c95ce381d39a963241c1e99abf71a7cb6b7243159631c2c99e83e9daa9b210fc0ef5151dd954a364d09f68db3042d1e0fcabe70033e986e99e2a4686554572815ff84ab6f18ebc81c530088487797dc067684e07b9278a7a4a167c451a6e38c99724ff00303f3ad35b19b8ea2c7be725449246a720aa4673d7d474fe5f5a91a18967043a4f2be3202719c7cdc64f1514b15dccbf34ece01e91becc1f4036ed61f53525ba148db36972ee0821a2c7047d700f04f714585b9699235ff58021520608e01c7afa54b6fb269308012371fbc7046338fcc5509dd4c4a30aa81f1f27ca08c7a7a75a98db12405590978d65ddd47f10eb9f6ac9ec64a6d6c5ab7977806340cab824676fe1cf7ab0371bb914132003ef67927f2fad554389a5930a4329f330410304723f9fe7566228b0cb9b6f31a390344f93c0603d3a8c63db9350c972b935a4cd2c2b0c8b96dfbf6aae0e40032493df1cf14d90c915cac65cff00acc2f038527953ff00d7a9223e65d449e4c49b9bcd99ca004a8232073c8c023a77aab792cdf692b2c632ccc436d0a4e4e464f53d4543dcaa964ae87f2ad192082a092e84f279ce4fa8ff00ebd112199e099491804365ba91c03f5e4f5aa8ec4b49129dca01cb6396dcb8fcf3dfeb51e9ac20dc71281b803b81dbb979073e9d47a13c51ba3965263a496268a742e77a92f002bc6f073b49edbb91ed806acc335e49624a656d94073f37cb83e9ebce067deb3d2ce792f040372cee0ca3ba94c677e7b0e3f9f151ca82eae56dd599c750db49ce38e9e9ce6b5490d49a638807cc468de47279da303a7e3ebd691d3ca8e3f2e1925da3a1c0393dba53d6c840e159e34504e1246505bdf03241aa6c62724094961c602f51daa92b95cd1d8b42e1c440282abe6052bc9391ce7f020522dfc8eb22843b18eee0679ce7249fa556b841f64403719379624719edfd3f4a8614972158bac79e486ea7f9fad55b42b5e85a7ba69812e0b1278e01c7a0ef4192ea1116495597e6556da372f4cf1d01c67f0a6a009322b8de54ee09bb01989381f8719fcbd695d248e65ff489190f2fb860e7e9d3f0cf6a342a367ab1ea9f689cb16f91bb03c91c7ff5eb434f68162963577cb8053e60482083c03df04d555768e38c390c5872e83820f20fd7b5580e80c56e8f977e857b7382693368c90d79e25bc8e558a46746dca5d5140e3d7aff002eb8ab572c2d26f2e2c0fbc1550606c2aa57ff00af9ee2a3482dcc4c5e52e48c61db69f53927a0f6ff00f5d1b2592279b70778949c8c1f940e08c7b5672f222aabea8a2d74cf664101956d8c88aa724b29e71f80fd69f15c2ca1446bb8608519e495038c63d2a65b6887ee9319663228f671cafd3233f8d361b755b34280314c0e48ce0e073f5feb49c93328bbb1c678d5dd1bcb233920f1f4c63a0eb5a5a5cc279fc942e1003f2b72074ef58b25bb943bce7e71b1beeede41ff00eb7ffaeb67c3796d45d1583280c0807f8b8e9fd7de93d8de936a4a2ceae052a8832381d45737e3977b5d2e0c121ae26c0c0eca335d3db40d23ed51d8039e6b8ff8813093534b2dc49b548dd57b36f247e18c66b6a51d1c99a579743939e322f265c9f91f6727d3a5685b5abcd1c821d892ae3877519fa03ce7f3ace8c99e4b94018bb13b4e70724f5fc335a36b67f689b08809419676195400e41f66fad36ce78ab8d6b5087ca6bab73bc8c0330f981e9c0e71ff00d6aaee0091f7e42173ec40edfcbf4abd1da192232b856f31ce772e471fd3b54242bb3066195e7a673d864d4a1d9119091c853072724e3b67a9fd6afce915ddf4c872ab3313f29fbbc718cfeb4cb2b091d3cd0a4b6e0140c63241c73fe7b542e8d049b536b10bbf7b3e157d724fd39a3d06e3a6876fe1d9d2e349457e6480f96ebdc0ec7fcfa56d472b2c8b13aa885be524f4615e736b7d3699a925d42b840c52640d8de8791f538c918f4fad7a3b583ea7a624314ed10701965419f7c8aeaa52f74574cf3ad6ac134ad66ea08c9f2cb6f468caf1f5c82323dc547a75cbc77db8344ca07023c86c671f32f6fc383f8d743e22f08cd6fa5497892cd35c40db9d8f5743c1e3dbafd335c95b3bfda11f25518953c0c03d8fbfff005eb1a917a9ac2ea67a2c0cb25aaba939e73ed57a0999a1f94fcc3a81ce6b134999da1da7961dbd6b4619423b104839e86b18e874c95d1ce78c3762d997251b7f23b1e31fceb2b4899becdb3258a1da4fad5df19cbbdace246c105e52476e401fd7f2ac8d2ee76c8f1724850dcfd4e6bb7ed2381bd4ea2d65931891414231b81e56aaf8aeec2e9cb1c2e15b2bf3119c54ba75c24afe51e090719acef15af971dac646773ee273e828ab7d0d29b4936cccd175330dc344cc40380063be2ba883514688a1ebe9ed5c0af965c104e4b71cd695bde4898527383ce326b0947b0e13496a6c5e244fca9e7af5eb5932b84e7e5007534ad7a82438271804f1d2a39a74b8fdd22e0b0ea4f1f8fe359b8b6cdb9d2206b951346e0e416e84532797cc95bcb95b69194563818383fd2a016ec222922b7ccc08a8e44da422be08c6013ce7bd6a8e794a499a0934c400497df82c4b671ee3279c71439466667cc6770da78c63a1c9ebfa545680998f9871b173c9edc62812ca5fb3bb8cc849c73f5a0b826d5cd1672f2b9f2e2876120151bbb762dfe79ab32159121662e5bcb08a0b61481ce481d09ddfe7155208c3293201225b9de403c33f651ea3dfeb5312b23c70bb9796442e463041e36f3ee013f97ad65239799daccb56f1a5c30458cc1128c3004bf5fcb3d71e9f9548844110584008a815031e303eef3dfb5504ba96772a5a216eedb4c6c7696031d87b55eb3859609848ede563cc0b8ced39c718ebcf1f5cd44d5b72ad6d4ba00919d55cb33844472a41424e473dc671f91aa5731adc10d8f9cbe491cf0471f4c7f9e9538b8c5a8dccb1c0854a6e3cbb8f53d80038fafbd416ce86288aa64871c1e40cf4cfebf9542d55c7cda2bec559999edde70c098e40519072a063230073c1fd7bd24727f66055d91b5de70ab210c212304123a16e981d17a9e78abd6969f24576aa23565c45ba4e02f27713dcf1c000fad668b6b5b4b868648e6b991e360412235dac430fef1ce40e78efeb571481c5363564b8bab8f2a7924706419f5e46e6fcf69cf6ef509c36a05d1b6a12d824e49ce38fc8fe95aaff0067937469e646c31b423e723a60f00f7f5fc6aa49a70f99ad9c384dbb5b9573c63254f63c7427a53464e0ccb27ca452222000000536718c74f4ea2a4862f36766842b36f3b474638f6cf3d3b55e0be627ef94cb0c58ca29002bf3c0fa9fe46aa3daf9688c0b025c919e4823b0abbf506bb12796b2465846abf20cb30e09279fe74ed3a25dc4cca8cabb9c93c1f94123e9c8e94e725d64c82ccb228dca3247a1fcc0fca99109199b318642482cb9efea3eb49b6d3358d9a1638d5aea256e8cc5f2474653c7e7c8a6ac8a9724042e4391b9f8079e9d39e943030ca158a82dc80483c67b7e7cd5858f748a8268c063b3774f98f6391c7a509fba4a5a1042e7eccd215db1a8f302a818cfa536dcba66660c5d55b9ef920d5b8d4af9caab22cc01cb705b70c1231d3b1eb4e91bce9644de1800012898e47d7d39e6a54b7348c77d48edadcdf869101deb805645cac80f3d4f7edfad5bb16593cc8e450aad131c26770f9718cd57888687fd163c9190179c0efc9abc461ad2f1a37dd244af2c7cf218119cfd5581ef91ef5337a3b152d15910088bace372f2a140f56ced18fa74fc6a3b681e3b39a27051a428cc48f4c951cff9e7bd4a88b169b3b1c2323003b7f129078fa9ab11472f9817ccc8dd88ced20ec5e0291f9e3d9876c56517a19422f731e2699dc0899c46abb9571ce0f739fae71f4adff07592c5a93624665116e0ecb8c8c81f87d2b3b4d8d24d4511a35746625c76002e4e493d78c73ed5d7786ace6874b6bfb98e20251bd428c828ab8fcb39c7b62b596a74416a6f69f98a3f370adf3139f5c1e95e73e2991af7c63a8f980232c712fb001411fcebd33468c085a25c98cbb6371e0640c62bcdf5c0f2789751959704c8325871c20c8ffc76b572b5333a9dcc48e282040d2a867c00590ee4e07407a938f602ac4d24ad68bb24d824504c6a06d43cf6c73c8efdaaddd59c37021732220507290e01dec4673db1c0c7b67150984c70bbb365536818c0e79cff004a8e6b97049a1f7320860821403693858f03e7cf27f1eb50337990c6648419225d9800642f3cf4c70703e87f1a859b7dd44d20e3cadb92b9e99078ef8cd593f678d233e634aa3eec8ebb88ebc1e692d85676d464b793bba46b8e9942b9e0fb73fa1a86e249a464120dedc10b9e99ee2a6558bcc6266f908c139c0600f43eb4d31c4de52c7210d21007ca558fa739fad5a7a14eca04db5134eba668d4ccb1a824be4f0780074e4171f80aeb3c09ad4634636b75388e2b36d8824601ca1c10063f841c807db1dab8db79dae2d9a158d4adc4722a6d3c65795c1f4201fc7ea6ac68af1c5a8ae1486f908c7073bb9fae45694e5ca6095b73d79e78645562e36b63af7af25f12787868d7ced669b6c6539888c908dfdc23dba8f6e3b5750be26b5db36e0eb144d988ec3f30c76fc735cfcb7f04ba5de47333b4b79fbc88e78460e46ef5e98e3bd6974cdaeb41da35f092d55812b229c118f4e0d6ec12c77237a9e73835c4e9339b4d45a19258d91ce148e30dff00d7aea726de44b88f9524798a3f9d62e1d8d613d4c5f1ac3e56a76df783496c769c7040639fc79158d6f22da5e5ab7cc55a3556fc7a9fe55d678ea1f374eb1d4a161fb993631c7f0be31ff8f2afe75c3cd2b365f181c119e7a56cf4699c924d3b9d8c2a126475eaac318aade32f9ed2c240b93965c0fc2a789b7431ccb9c3283f9d37c44be66881ca06549001f8f1fcc0adea2f742f7471415430628db81f9463a1ab424608a06d0e376e61ebd73e84d439fddb00bb777fb47903fcff004a045e6a17299c636f1dfbe2b06427a933ed9959b7e49feef038a75bc126d2f0b1214fcdb8609ff22abe55edc84186525863f223f3c5363999338cb3e72be87f1a4f4358b45b40ed2aee1953c818233537916eebfead15c37cad8c13ed9aa71caf190ac5770c0653d57dfad592c308e4e18838c0ef9e9458b56d8530aed990823036938f4edf8fad31a3cc0b21c16278047423b8a71b83316331385c6589ebed53b5dc90ab339546dd85455c7f9fe9458abd8b69730ba3c76ae6652872ae3616c9c93823a9c63afd2a085a77324f70cd1ccc410a1483838fcba8fc054932a0b36895942b107e6c81839e08a6bda1b5b63000091f7d99faff009cfe9ef58e8ce49449ed7ce6b894c02244688bb7cc000e7be71f2afde39ea2accd792dcd9b436f930c488aaf2a6cc91c9723b0e471d8673ce6a8471bc36ef6d64ac51d4493dd3aec04ff007467a01c9ea4e4d4970c15e3f2660ecd1808ea8ca59b008cab01d80a4d5cb92b46ecb379209990e59ed906101e01c7258fb9393f434595cb34925ac2fb1a489ce14e013ce327e991d3eb55e085eeb4eb8deafe747b65407ab8fe3181d323903da9d64be65e4732053379cc361cf5c9c7a76feb4ac849b72b167ed4b21003c920f954023040cf63db8eb8c54900b3927691da48550fca492fbb8c74033e9dcf155a79f33b44a4b02fb433752001dfdffad56631630ed22aaa83190990c49c9279cfd3b718a7ca529b5a22ec76d0c84bac9146c728dba7c9e9c6411b40c8eb9ef53dcda5d4f74f3b5b3141897cd491088fa03c06c80703181d6a842e16385629863049563d7071cfe3e948f1a2324aa8867fbf964cf5c639ed9e2a5756549db4b166f184ecb1ca18a86f9724a9627bb63a8e833d78c679351fd9e02a824b8c2a0393b58e71d4f19a89a785a449111b78002a1fbca7827f2e40c7a8f4a7dac721679590ec8c10dc9c6483c7e429ad37236760607ca45255d4bb063b7d863b6702a155b6c2b997cc323636085be53ea1b760fe552206106ce484724b75c67a1a644a258b7c2a11437276e48e39279f7cfe18a498256d42296d92d587d9d995801cbed19cfe3ee69c156e048bb4a203f20cfca07e5ef421408a238d9932700c9cb2e3249c7d0d4e891991258f2bb8af98ac4fca7238e47b7b7f4a6dab169d91098cc6acd210db180624f7ce33f5e9f953a09a4f2847fbd9b712fb2350c7008247381e9dfd6accaeda8c2d02fef268d8940385750d9f2f18ea36f1f88ee0d49a5d989378775865907c88cdc7b2e474273c7e03bd67e6c508f34868620468acf24587ddbfae3231900f3d49c8f4a9a74287cac301191b9c7200fbdec73bbf9d395e0b7957722304528731e47ccc72083d3ee9e3daab2876799dbce4691301891b70390401ce7b527b9a349bb22e0844f62f6f98c93186dcde81b824fd7afa5743a3f8466bcb5be93edc88d13b3a47b46548ee49e31d7a7af5ae7ad4afd9d4161e6163903e43ce3031dfa7f3adcd3fc4f3d9c12c4d12b99d177b34981190083818c1c8c75ef453b293b8ecd3d0bda6780a0de8d71a934cb29f3310200a50f25492739e6b5f5c92d6d6cdedf67cad0ac71041c6338c607a0ae7ed3c4a0456508b27748f0bbd7e566073dbfe043f5ef54bc49ad5b4f796e8adba2894b62342016fc3f83f539f4ebace4bec96afadc9ef3c557767a74b1696a24333ed13b7ca530307af1db02b9c92669619ae25937caedb9dfa02d9dad8f41c1fcfde926699c01263f76097c6447d000147b63fce6a5b88a4fb0431c64179c98c841907200c678ebc1fad60e4da51326afa10e9702bc2e8b19e509c449c60608caf63d7f2f7a65d155de84a0c30cc639c67d33fe7e98c506268de48405322315c6580623b647518ebf407b522a797010511b3f3150bf8647f9f4ab4cb4ad120dea182ba866da40da3903193fd2a29ad5d6c919c04f2e23bd4740c4f23f018fccd5c89d26654647054a80106d0bedc741d783503b2b09532549c86f94637608ce3f2fd7d29a6f616add994e16c9da2342992406040240ef8e79ff0a5925da3312fce8772e46067a77e4633f954b2dbfcc53cbddf3643a8233ef93d0f5aab1891e505ddb69fddb8c671ef5a742669c572930630ac1226d57898b0dad9fa8ce3be7f5a49d0c666f28b0c39298eb82a3fa11492b318240cbb65572320f53c64fe869b7a1c9b20ec1898d7183fc41981fae30b4448d64c6b00564904331230a1dc1c151d703af5cf34f94cad6d6b899895472db495e4b938c0ff003c511ce63b76491a7752bf32348cde5f7ea0d412ed9d6139742bf75b6e41cf3d7ad69d4dd28a209ad8a4aaed2839e8021e3ebc5753a05f9b888c329dd228ef9f9bdeb0b32b380d3465707876ea3dbbe68d2ee85b6a11b00065b6923be6a96a81a4968770238ef349b9d3e4380ea5067b67953f81af3892de45768e6511b29c3a939c63838fe55dec4e60bb07b1e0e79ac2f165834730d42341b25c090f60dd8fe231f955dd344f2dc724fe5d9c110ce1555738f4ad18e25bed1e6b7918849508047553d88fc6b3add7302abf276feb8a826bc9eded0c51e4e78c77a7527b2269c559dce78aa841bd77903eea123a7279a96693cbb7b60d1e0b967c6f20819c0fe44fe548026e5524063c9cf071dff4a2467b99805da59540dbdf1ce38ed50643124f9826d23239f7ff003c54f6eb14d202432b16e71d0f73f4a855544d8970cabc150704fe3daa687c9918e36c0aa8c46c058e3ea7bfb9a13ea1b6c396252dbd5406727631ea07735192872b21511af0aaadc81efefde9c93fda2638555539c1c927a71509890920b36d0c78033b8e718c50c2e1e664946f296307393f7c1c74cff4a9d944b8c98c15fe173823bf07d7f2a81cb26d42d19c8e1193a73c75efe95217dd23070373fcc194707de932d4cd28f2b1f9a09d8471938f9bd3dfdbf3a9ae8992c5e55da0e4c2641d588c607fe3d572c6da5bc924594c4df29dbf28183c91d07eb59f266284c32c6ae81b002b95e49e4ffe3b58adf5172b24406c52247523389361ca8e9db047a673cd5b8f5211dfb4cd1c436af1fbde482318e47b8c1cd3adf53580963631cca771c4b29f9463242f1e8b81f5cd3c6a922dac896b6b6f0dc9983098139557e8a0f60011db922a5bd762df2a45732cb691da3dab4090440c8b6f1387f2d89c1ddc92ccc3ab1ede9ce5ec8bf6990c4254552c159887215870477e324639e94896df6a81946d8cbc8aaaa8301719e727273c81f40299726382dae27932f2dbca178500070e5372fa7afe1db9a7d6e2715277249e389a3b39e490851185e173e63670581f4c71ee6b3842f221603181f2fbfb0fc2ad79df6865380087c12c0b95c10a0282700739e9ebeb5627896e618edc168e389d812a7971d47d0e47bd3b8a50e5f78aaea8b6f0b17760e7680a465bab631df8cf4f4a56fde4eae5ce33b8e4e32a98cf7c761533bfc8e8859760e9ea4903af5ef552ebc96baf2774be61895b6e06c58ce71ee5b3c9e82886a8126c6c36fe55cb3875f2db2bb4027ebf4ab9a68335d3c2e3ca33066ddc801f6ed07f2e3f4a843182d9228cfcb2650ee51d0714d1119bcd57628d226e6319c6072001f4c1ff00228b5d32ec88e68e4b79a650332aae1474c907047f3fcaa66fdcc6a15f8f981e7b9c64e3bff854768802797960aaeca39c923df3ee01a7cc504dc29001d99079ce79fc38ff00f554b466e2d683e351f688ca47211b5937b67e5078e71c7d29fa7c25a5fb99999c1c86e49ce0e0fb53ad6de776967f3142c4594819c9ec3fad5e8ada5894dded8140608db14ab1dd8fc3f4a1fc26b04a5a75b1520984cf238017ca5795a5203e5460f00739e9c77a6de9b76b937001d8181c2e4f96de981f9e7ad4e924423b9ba40cbb959766011d547f51f95208b6dc9b757c890feef09b026def8c9c1e71f4a8689947949ef209b5a941827492554fdeec059a40a325b68e7791f8f39c1e699146182ac321903f0aeb21e78c2b607193c73ed4ed3c12a594045490fef1787249cff3e782318144a6448c0f913e56127963ef6319fce93ec8d3952d4b49736d6f98e449d9492ad20c160c3ebd467ffad4cb7b29865bce8da2ce1300607b7b606475a8ada4cde431948c9018a820952473f30cf352453496697851632ecca76e303a0ff3d297a0d4ef2b96e52b245be174f3a255f359131804f0e3bf7c1f4383deabc5639656f34796324151f281c1ddf5e31d0d588ef16cefc7951229505b0147390036718ea38ce29f71a718af6f54dccac2d59d54939255b017f1007ea69db42de9a90dc476bb152293ccf371f391f273d7000ea464f27231c0a6c6b25b2a4cc559a243304753cf040623f0c81ed44109657b88e46000c90c32493c75a847ef26dd3fcc6653903a638c0c7a600151e8251bb1f1d933178cda14e431685830c6074ddd474e9e86b3e79245f33cddbc65369edd80c7af1f975ab3335bb29b8b51344b80bb4c878c13d39f63f5e0d57fb4dc4770ca93b9644cc81ce41e33c1eb54932aa47a08b2aba2853838e838e71ed552756f3480779cef382464f1c7b0e9533197cf0a188d8a06371c64007f2a90d825ada19a63995d0fce839e7ebf51f95688c6fcda3e8539a77410e6361b506460b2907b91d33cff002f4a67971a92c43942a06c76ce083d4e7a934e59622048902e402b96e78fa74ed556695ae1c83820b018c607e42aca96a4ee0c8a87681bdf2c00c9e801aab712cacb0040a1d1a4521881b43104f2071ce6a5990948f71e541ce3bd56936c4b0fcb8030e02fbf14e3b192d1123394908616f80060a3673c71fcaa0b8632328913183976dbd3d7be29f24d8653ce5816fd7153452eecc251588c659bb0f6f5fc6ad093e6656936a4382acc17919201feb4e1f67521d581dac03028cacb9e7af43d3b1a92628257891022962463b0cff3aa7bd892b9c8c96e7d6a9772deda1d95cdd24b1a35bbe64001c0ef55f55be9750d11e055c1f3235638e9f30aad667f711c981b90ed3ee3aff5adb8e3592da7650064739a88bbc8d7451b99518528403cf6ac9d59dfcb8a2403032c73efd2b4af02c48046319eb5cfde4a64d43ca75cafdd277107fcf4ab7ac9916f7445758a36455323c8a3217aedcf427a0191da9ad3322ed214679217807ea4f5a16756533c917caac3015c8393f4c7152816f326e7b541bd8a9209c9f4cd5194a05066c9c7978f4c7f9fd6ac0809b7e15155c7fac638c8cf6ef520b7f218b46aaa0138c120d3d8dba2a936d832654b236093d89e39a56279482da0314c8ee331ae73b3b0c7f9fcea18e2df8698ae14f2b93c1ebc1ab9226e8f7c6c762f387ebf9d565906e66545dca08424743ebfa51b6a3e556d09b2c64d93962ea4b0751c8cf61eb807f4a0452423663288770cf407d47b52a21b8b82548554e791cf03afd6a0b896104c5187c271b9b1934ae45ac7ffd9"

        image1 = hex_to_image(image1_hex)
        image2 = hex_to_image(image2_hex)

        text = (
            "The left image contains twice the number of dogs as the right image, and at least two dogs in total are"
            "standing."
        )
        encoding_1 = processor(image1, text, return_tensors="ms")
        encoding_2 = processor(image2, text, return_tensors="ms")

        pixel_values = mindspore.ops.stack([encoding_1.pixel_values, encoding_2.pixel_values], axis=1)

        # forward pass
        outputs = model(
            input_ids=encoding_1.input_ids,
            pixel_values=pixel_values,
        )

        # verify the logits
        expected_shape = mindspore.ops.shape(mindspore.Tensor(np.ones(shape=[1, 2]), mindspore.float32))
        self.assertEqual(outputs.logits.shape, expected_shape)

        is_pillow_less_than_9 = version.parse(PIL.__version__) < version.parse("9.0.0")

        if is_pillow_less_than_9:
            expected_slice = mindspore.tensor(
                [-2.4013, 2.9342],
            )
        else:
            expected_slice = mindspore.tensor(
                [-2.3713, 2.9168],
            )
        logits_np = outputs.logits[0, :3].asnumpy()
        expected_slice_np =expected_slice.asnumpy()

        self.assertTrue(np.allclose(logits_np, expected_slice_np, atol=8))
