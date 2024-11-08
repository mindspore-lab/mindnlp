
import unittest

import unittest
import numpy as np
from mindnlp.transformers import Data2VecVisionConfig
from mindnlp.utils import cached_property
from mindnlp.utils.testing_utils import (
    TestCasePlus,
    is_mindspore_available,
    is_vision_available,
    require_mindspore,
    slow, require_vision,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor



if is_mindspore_available():
    import mindspore

    from mindnlp.core import nn, ops

    from mindnlp.transformers import (
        Data2VecVisionForImageClassification,
        Data2VecVisionForSemanticSegmentation,
        Data2VecVisionModel,
    )
    from mindnlp.transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES


if is_vision_available():
    from PIL import Image

    from mindnlp.transformers import BeitImageProcessor


class Data2VecVisionModelTester:
    def __init__(
        self,
        parent,
        vocab_size=100,
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
        num_labels=3,
        scope=None,
        out_indices=[0, 1, 2, 3],
    ):
        self.parent = parent
        self.vocab_size = 100
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
        self.out_indices = out_indices
        self.num_labels = num_labels

        # in BeiT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            pixel_labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels, pixel_labels

    def get_config(self):
        return Data2VecVisionConfig(
            vocab_size=self.vocab_size,
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
            is_decoder=False,
            initializer_range=self.initializer_range,
            out_indices=self.out_indices,
        )

    def create_and_check_model(self, config, pixel_values, labels, pixel_labels):
        model = Data2VecVisionModel(config=config)

        model.set_train(False)
        result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.type_sequence_label_size
        model = Data2VecVisionForImageClassification(config)

        model.set_train(False)
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def create_and_check_for_image_segmentation(self, config, pixel_values, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = Data2VecVisionForSemanticSegmentation(config)

        model.set_train(False)
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_mindspore
class Data2VecVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Data2VecVision does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (Data2VecVisionModel, Data2VecVisionForImageClassification, Data2VecVisionForSemanticSegmentation)
        if is_mindspore_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "image-feature-extraction": Data2VecVisionModel,
            "image-classification": Data2VecVisionForImageClassification,
            "image-segmentation": Data2VecVisionForSemanticSegmentation,
        }
        if is_mindspore_available()
        else {}
    )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Data2VecVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Data2VecVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Data2VecVision does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass


    @unittest.skip(
        reason="Data2VecVision has some layers using `add_module` which doesn't work well with `nn.DataParallel`"
    )
    def test_multi_gpu_data_parallel_forward(self):
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

    def test_for_image_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_segmentation(*config_and_inputs)

    def test_training(self):
        if not self.model_tester.is_training:
            self.skipTest(reason="model_tester.is_training is set to False")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class.__name__ in MODEL_MAPPING_NAMES.values():
                continue

            model = model_class(config)

            model.set_train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss




    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # we skip lambda parameters as these require special initial values
                # determined by config.layer_scale_init_value
                if "lambda" in name:
                    continue
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )


    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "facebook/data2vec-vision-base-ft1k"
        model = Data2VecVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


import pathlib
import os
import inspect
def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    # this function caller's __file__
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    return tests_dir


# We will verify our results on an image of cute cats
def prepare_img():
    fixtures_path = pathlib.Path(get_tests_dir()) / 'fixtures/tests_samples/COCO'
    image = Image.open(fixtures_path / "000000039769.png")
    return image



@require_vision
class Data2VecVisionModelIntegrationTest(unittest.TestCase):

    @cached_property
    def default_image_processor(self):
        return (
            BeitImageProcessor.from_pretrained("facebook/data2vec-vision-base-ft1k") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head_imagenet_1k(self):
        model = Data2VecVisionForImageClassification.from_pretrained("facebook/data2vec-vision-base-ft1k")

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="ms")

        # forward pass

        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = (1, 1000)
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = mindspore.tensor([0.3277, -0.1395, 0.0911])

        self.assertTrue(ops.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_top2 = [model.config.label2id[i] for i in ["remote control, remote", "tabby, tabby cat"]]
        #print(type(logits[0].topk(2)))
        self.assertEqual(logits[0].topk(2)[1].tolist(), expected_top2)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        model_name = "facebook/data2vec-vision-base-ft1k"
        model = Data2VecVisionModel.from_pretrained(model_name, **{"use_absolute_position_embeddings": True})

        image = prepare_img()
        processor = BeitImageProcessor.from_pretrained("facebook/data2vec-vision-base-ft1k")
        inputs = processor(images=image, return_tensors="ms", size={"height": 480, "width": 480})
        pixel_values = inputs.pixel_values

        # with interpolate_pos_encoding being False an exception should be raised with higher resolution
        # images than what the model supports.
        self.assertFalse(processor.do_center_crop)

        '''with self.assertRaises(ValueError, msg="doesn't match model"):
            model(pixel_values, interpolate_pos_encoding=False)'''

        # with interpolate_pos_encoding being True the model should process the higher resolution image
        # successfully and produce the expected output.

        outputs = model(pixel_values, interpolate_pos_encoding=True)

        expected_shape = (1, 1801, 768)
        self.assertEqual(outputs.last_hidden_state.shape, expected_shape)