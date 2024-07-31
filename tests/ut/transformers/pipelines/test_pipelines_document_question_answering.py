# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import pytest

from mindnlp.transformers import MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING, pipeline, AutoTokenizer
from mindnlp.transformers.pipelines.document_question_answering import apply_tesseract

from mindnlp.utils.testing_utils import is_pipeline_test, require_vision, slow, \
    nested_simplify, require_pytesseract

from mindnlp.utils import is_vision_available, require_mindspore

from .test_pipelines_common import ANY

if is_vision_available():
    from PIL import Image

    from mindnlp.transformers.image_utils import load_image
else:

    class Image:
        @staticmethod
        def open(*args, **kwargs):
            pass


    def load_image(_):
        return None

# This is a pinned image from a specific revision of a document question answering space, hosted by HuggingFace,
# so we can expect it to be available.
INVOICE_URL = (
    "https://hf.co/spaces/impira/docquery/resolve/2f6c96314dc84dfda62d40de9da55f2f5165d403/invoice.png"
)


@is_pipeline_test
@require_mindspore
@require_vision
class DocumentQuestionAnsweringPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING

    @require_pytesseract
    @require_vision
    def get_test_pipeline(self, model, tokenizer, processor):
        dqa_pipeline = pipeline(
            "document-question-answering", model=model, tokenizer=tokenizer, image_processor=processor
        )

        image = "INVOICE_URL"
        word_boxes = list(zip(*apply_tesseract(load_image(image), None, "")))
        question = "What is the placebo?"
        examples = [
            {
                "image": load_image(image),
                "question": question,
            },
            {
                "image": image,
                "question": question,
            },
            {
                "image": image,
                "question": question,
                "word_boxes": word_boxes,
            },
        ]
        return dqa_pipeline, examples

    def run_pipeline_test(self, dqa_pipeline, examples):
        outputs = dqa_pipeline(examples, top_k=2)
        self.assertEqual(
            outputs,
            [
                [
                    {"score": ANY(float), "answer": ANY(str), "start": ANY(int), "end": ANY(int)},
                    {"score": ANY(float), "answer": ANY(str), "start": ANY(int), "end": ANY(int)},
                ]
            ]
            * 3,
        )

    @require_mindspore
    @require_pytesseract
    @pytest.mark.skip
    def test_small_model_ms(self):
        dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-layoutlmv2")
        image = INVOICE_URL
        question = "How many cats are there?"

        expected_output = [
            {"score": 0.0001, "answer": "oy 2312/2019", "start": 38, "end": 39},
            {"score": 0.0001, "answer": "oy 2312/2019 DUE", "start": 38, "end": 40},
        ]
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), expected_output)

        # This image does not detect ANY text in it, meaning layoutlmv2 should fail.
        # Empty answer probably
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(outputs, [])

        # We can optionnally pass directly the words and bounding boxes
        image = "./tests/fixtures/tests_samples/COCO/000000039769.png"
        words = []
        boxes = []
        outputs = dqa_pipeline(image=image, question=question, words=words, boxes=boxes, top_k=2)
        self.assertEqual(outputs, [])

    @slow
    @require_mindspore
    @require_pytesseract
    def test_large_model(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
            revision="9977165",
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9944, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0009, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9944, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0009, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9944, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.0009, "answer": "us-001", "start": 16, "end": 16},
                ],
            ]
            * 2,
        )

    @slow
    @require_mindspore
    @require_pytesseract
    def test_large_model_chunk(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
            revision="9977165",
            max_seq_len=50,
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9974, "answer": "1110212019", "start": 23, "end": 23},
                {"score": 0.9948, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9974, "answer": "1110212019", "start": 23, "end": 23},
                {"score": 0.9948, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9974, "answer": "1110212019", "start": 23, "end": 23},
                    {"score": 0.9948, "answer": "us-001", "start": 16, "end": 16},
                ]
            ]
            * 2,
        )

    @slow
    @require_mindspore
    @require_pytesseract
    @require_vision
    def test_large_model_layoutlm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "impira/layoutlm-document-qa", revision="3dc6de3", add_prefix_space=True
        )
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            tokenizer=tokenizer,
            revision="3dc6de3",
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

        outputs = dqa_pipeline({"image": image, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
                ]
            ]
            * 2,
        )

        word_boxes = list(zip(*apply_tesseract(load_image(image), None, "")))

        # This model should also work if `image` is set to None
        outputs = dqa_pipeline({"image": None, "word_boxes": word_boxes, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.4251, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.0819, "answer": "1110212019", "start": 23, "end": 23},
            ],
        )

    @slow
    @require_mindspore
    @require_pytesseract
    @require_vision
    def test_large_model_layoutlm_chunk(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "impira/layoutlm-document-qa", revision="3dc6de3", add_prefix_space=True
        )
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="impira/layoutlm-document-qa",
            tokenizer=tokenizer,
            revision="3dc6de3",
            max_seq_len=50,
        )
        image = INVOICE_URL
        question = "What is the invoice number?"

        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

        outputs = dqa_pipeline(
            [{"image": image, "question": question}, {"image": image, "question": question}], top_k=2
        )
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                [
                    {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                    {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
                ]
            ]
            * 2,
        )

        word_boxes = list(zip(*apply_tesseract(load_image(image), None, "")))

        # This model should also work if `image` is set to None
        outputs = dqa_pipeline({"image": None, "word_boxes": word_boxes, "question": question}, top_k=2)
        self.assertEqual(
            nested_simplify(outputs, decimals=4),
            [
                {"score": 0.9999, "answer": "us-001", "start": 16, "end": 16},
                {"score": 0.9998, "answer": "us-001", "start": 16, "end": 16},
            ],
        )

    @slow
    @require_mindspore
    def test_large_model_donut(self):
        dqa_pipeline = pipeline(
            "document-question-answering",
            model="naver-clova-ix/donut-base-finetuned-docvqa",
            tokenizer=AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa"),
            feature_extractor="naver-clova-ix/donut-base-finetuned-docvqa",
        )

        image = INVOICE_URL
        question = "What is the invoice number?"
        outputs = dqa_pipeline(image=image, question=question, top_k=2)
        self.assertEqual(nested_simplify(outputs, decimals=4), [{"answer": "us-001"}])