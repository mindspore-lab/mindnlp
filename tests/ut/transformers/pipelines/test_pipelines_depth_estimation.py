# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import unittest

from huggingface_hub import DepthEstimationOutput
from huggingface_hub.utils import insecure_hashlib

from mindnlp.transformers import MODEL_FOR_DEPTH_ESTIMATION_MAPPING
from mindnlp.utils import is_mindspore_available, is_vision_available
from mindnlp.transformers.pipelines import DepthEstimationPipeline, pipeline
from mindnlp.utils.testing_utils import (
    is_pipeline_test,
    nested_simplify,
    require_mindspore,
    require_vision,
    slow,
)

from .test_pipelines_common import ANY


if is_mindspore_available():
    import mindspore
    from mindnlp.core import ops

if is_vision_available():
    from PIL import Image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


def hashimage(image: Image) -> str:
    m = insecure_hashlib.md5(image.tobytes())
    return m.hexdigest()


@is_pipeline_test
@require_vision
# @require_timm
@require_mindspore
class DepthEstimationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING

    def get_test_pipeline(
        self,
        model,
        tokenizer=None,
        image_processor=None,
        feature_extractor=None,
        processor=None,
        torch_dtype="float32",
    ):
        depth_estimator = DepthEstimationPipeline(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            torch_dtype=torch_dtype,
        )
        return depth_estimator, [
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
            "./tests/fixtures/tests_samples/COCO/000000039769.png",
        ]

    def run_pipeline_test(self, depth_estimator, examples):
        outputs = depth_estimator("./tests/fixtures/tests_samples/COCO/000000039769.png")
        self.assertEqual({"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)}, outputs)
        import datasets

        # we use revision="refs/pr/1" until the PR is merged
        # https://hf.co/datasets/hf-internal-testing/fixtures_image_utils/discussions/1
        dataset = datasets.load_dataset("hf-internal-testing/fixtures_image_utils", split="test", revision="refs/pr/1")
        outputs = depth_estimator(
            [
                Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png"),
                "http://images.cocodataset.org/val2017/000000039769.jpg",
                # RGBA
                dataset[0]["image"],
                # LA
                dataset[1]["image"],
                # L
                dataset[2]["image"],
            ]
        )
        self.assertEqual(
            [
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
            ],
            outputs,
        )

    @slow
    @require_mindspore
    def test_large_model(self):
        model_id = "Intel/dpt-large"
        depth_estimator = pipeline("depth-estimation", model=model_id)
        outputs = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
        outputs["depth"] = hashimage(outputs["depth"])

        # This seems flaky.
        # self.assertEqual(outputs["depth"], "1a39394e282e9f3b0741a90b9f108977")
        self.assertEqual(nested_simplify(outputs["predicted_depth"].max().item()), 29.306)
        self.assertAlmostEqual(nested_simplify(outputs["predicted_depth"].min().item()), 2.662, 2)

    @require_mindspore
    def test_small_model(self):
        # This is highly irregular to have no small tests.
        self.skipTest(reason="There is not hf-internal-testing tiny model for either GLPN nor DPT")

    @require_mindspore
    def test_multiprocess(self):
        depth_estimator = pipeline(
            model="hf-internal-testing/tiny-random-DepthAnythingForDepthEstimation",
            num_workers=2,
        )
        outputs = depth_estimator(
            [
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
                "./tests/fixtures/tests_samples/COCO/000000039769.png",
            ]
        )
        self.assertEqual(
            [
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
                {"predicted_depth": ANY(mindspore.Tensor), "depth": ANY(Image.Image)},
            ],
            outputs,
        )
