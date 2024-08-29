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
"""Testing suite for the Mindspore ViTMSN model."""

import unittest
import numpy as np

from mindnlp.transformers import ViTMSNConfig
from mindnlp.utils.testing_utils import require_mindspore, require_vision, slow
from mindnlp.utils import cached_property, is_mindspore_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor



if is_mindspore_available():
    import mindspore as ms
    from mindspore import nn, ops

    from mindnlp.transformers import ViTMSNForImageClassification, ViTMSNModel


if is_vision_available():
    from PIL import Image

    from mindnlp.transformers import ViTImageProcessor


class ViTMSNModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.attn_implementation = attn_implementation

        # in ViT MSN, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return ViTMSNConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            attn_implementation=self.attn_implementation,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = ViTMSNModel(config=config)
        model.set_train(False)
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = ViTMSNForImageClassification(config)
        model.set_train(False)
        result = model(pixel_values, labels=labels)
        print("Pixel and labels shape: {pixel_values.shape}, {labels.shape}")
        print("Labels: {labels}")
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = ViTMSNForImageClassification(config)
        model.set_train(False)

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_mindspore
class ViTMSNModelTest(ModelTesterMixin,  unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViTMSN does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ViTMSNForImageClassification,) if is_mindspore_available() else ()

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ViTMSNModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTMSNConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ViTMSN does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/vit-msn-small"
        model = ViTMSNModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_mindspore
@require_vision
class ViTMSNModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return ViTImageProcessor.from_pretrained("facebook/vit-msn-small") if is_vision_available() else None

    @slow
    @unittest.skip(reason="The random number generation of pytorch and mindspore is inconsistent")
    def test_inference_image_classification_head(self):
        ms.set_seed(2)
        model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-small")

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="ms")

        # forward pass
        outputs = model(**inputs)

        # verify the logits
        expected_shape =(1, 1000)
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = ms.tensor([0.5588, 0.6853, -0.5929])
        print(outputs.logits[0, :3].asnumpy())
        self.assertTrue(np.allclose(outputs.logits[0, :3].asnumpy(), expected_slice.asnumpy(), atol=1e-3))