# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import gc
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

import datasets
import numpy as np

from mindspore.dataset import GeneratorDataset

from mindnlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TextClassificationPipeline,
    pipeline,
)
from mindnlp.transformers.pipelines import PIPELINE_REGISTRY, get_task
from mindnlp.transformers.pipelines.base import Pipeline
from mindnlp.utils.testing_utils import (
    # TOKEN,
    # USER,
    CaptureLogger,
    RequestCounter,
    # backend_empty_cache,
    is_pipeline_test,
    # is_staging_test,
    nested_simplify,
    # require_tensorflow_probability,
    # require_tf,
    require_mindspore,
    # require_mindspore_accelerator,
    # require_mindspore_or_tf,
    slow,
    # torch_device,
)
from mindnlp.utils import direct_transformers_import, is_mindspore_available
from mindnlp.utils import logging as transformers_logging


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from ....test_module.custom_pipeline import PairClassificationPipeline  # noqa E402


logger = logging.getLogger(__name__)


PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent.parent.parent.parent, "mindnlp")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(PATH_TO_TRANSFORMERS)


class ANY:
    def __init__(self, *_types):
        self._types = _types

    def __eq__(self, other):
        return isinstance(other, self._types)

    def __repr__(self):
        return f"ANY({', '.join(_type.__name__ for _type in self._types)})"


@is_pipeline_test
class CommonPipelineTest(unittest.TestCase):
    @require_mindspore
    def test_pipeline_iteration(self):
        class MyDataset:
            data = [
                "This is a test",
                "This restaurant is great",
                "This restaurant is awful",
            ]

            def __len__(self):
                return 3

            def __getitem__(self, i):
                return self.data[i]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert"
        )
        dataset = GeneratorDataset(MyDataset(), column_names=['text'])
        for output in text_classifier(dataset):
            self.assertEqual(output, {"label": ANY(str), "score": ANY(float)})

    @require_mindspore
    def test_check_task_auto_inference(self):
        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")

        self.assertIsInstance(pipe, TextClassificationPipeline)

    @require_mindspore
    def test_pipeline_batch_size_global(self):
        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")
        self.assertEqual(pipe._batch_size, None)
        self.assertEqual(pipe._num_workers, None)

        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert", batch_size=2, num_workers=1)
        self.assertEqual(pipe._batch_size, 2)
        self.assertEqual(pipe._num_workers, 1)

    @require_mindspore
    def test_pipeline_pathlike(self):
        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")
        with tempfile.TemporaryDirectory() as d:
            pipe.save_pretrained(d)
            path = Path(d)
            newpipe = pipeline(task="text-classification", model=path)
        self.assertIsInstance(newpipe, TextClassificationPipeline)

    @require_mindspore
    def test_pipeline_override(self):
        class MyPipeline(TextClassificationPipeline):
            pass

        text_classifier = pipeline(model="hf-internal-testing/tiny-random-distilbert", pipeline_class=MyPipeline)

        self.assertIsInstance(text_classifier, MyPipeline)

    def test_check_task(self):
        task = get_task("gpt2")
        self.assertEqual(task, "text-generation")

        with self.assertRaises(RuntimeError):
            # Wrong framework
            get_task("espnet/siddhana_slurp_entity_asr_train_asr_conformer_raw_en_word_valid.acc.ave_10best")

    @require_mindspore
    def test_iterator_data(self):
        def data(n: int):
            for _ in range(n):
                yield "This is a test"

        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")

        results = []
        for out in pipe(data(10)):
            self.assertEqual(nested_simplify(out), {"label": "LABEL_0", "score": 0.504})
            results.append(out)
        self.assertEqual(len(results), 10)

        # When using multiple workers on streamable data it should still work
        # This will force using `num_workers=1` with a warning for now.
        results = []
        for out in pipe(data(10), num_workers=2):
            self.assertEqual(nested_simplify(out), {"label": "LABEL_0", "score": 0.504})
            results.append(out)
        self.assertEqual(len(results), 10)

    @require_mindspore
    def test_unbatch_attentions_hidden_states(self):
        model = DistilBertForSequenceClassification.from_pretrained(
            "hf-internal-testing/tiny-random-distilbert", output_hidden_states=True, output_attentions=True
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-distilbert")
        text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        # Used to throw an error because `hidden_states` are a tuple of tensors
        # instead of the expected tensor.
        outputs = text_classifier(["This is great !"] * 20, batch_size=32)
        self.assertEqual(len(outputs), 20)


@is_pipeline_test
class PipelineScikitCompatTest(unittest.TestCase):
    @require_mindspore
    def test_pipeline_predict(self):
        data = ["This is a test"]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert"
        )

        expected_output = [{"label": ANY(str), "score": ANY(float)}]
        actual_output = text_classifier.predict(data)
        self.assertEqual(expected_output, actual_output)

    @require_mindspore
    def test_pipeline_transform(self):
        data = ["This is a test"]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert"
        )

        expected_output = [{"label": ANY(str), "score": ANY(float)}]
        actual_output = text_classifier.transform(data)
        self.assertEqual(expected_output, actual_output)


@is_pipeline_test
class PipelineUtilsTest(unittest.TestCase):

    @slow
    @require_mindspore
    def test_load_default_pipelines(self):
        import mindspore
        from mindnlp.transformers.pipelines import SUPPORTED_TASKS

        set_seed_fn = lambda: mindspore.set_seed(0)  # noqa: E731
        for task in SUPPORTED_TASKS.keys():
            if task == "table-question-answering":
                # test table in seperate test due to more dependencies
                continue

            self.check_default_pipeline(task, "ms", set_seed_fn, self.check_models_equal_pt)

            # clean-up as much as possible GPU memory occupied by PyTorch
            gc.collect()


    # @slow
    # @require_mindspore
    # def test_load_default_pipelines_pt_table_qa(self):
    #     import torch

    #     set_seed_fn = lambda: torch.manual_seed(0)  # noqa: E731
    #     self.check_default_pipeline("table-question-answering", "pt", set_seed_fn, self.check_models_equal_pt)

    #     # clean-up as much as possible GPU memory occupied by PyTorch
    #     gc.collect()
    #     backend_empty_cache(torch_device)

    # @slow
    # @require_mindspore
    # @require_mindspore_accelerator
    # def test_pipeline_accelerator(self):
    #     pipe = pipeline("text-generation", device=torch_device)
    #     _ = pipe("Hello")

    # @slow
    # @require_mindspore
    # @require_mindspore_accelerator
    # def test_pipeline_accelerator_indexed(self):
    #     pipe = pipeline("text-generation", device=torch_device)
    #     _ = pipe("Hello")

    def check_default_pipeline(self, task, framework, set_seed_fn, check_models_equal_fn):
        from mindnlp.transformers.pipelines import SUPPORTED_TASKS, pipeline

        task_dict = SUPPORTED_TASKS[task]
        # test to compare pipeline to manually loading the respective model
        model = None
        relevant_auto_classes = task_dict[framework]

        if len(relevant_auto_classes) == 0:
            # task has no default
            logger.debug(f"{task} in {framework} has no default")
            return

        # by default use first class
        auto_model_cls = relevant_auto_classes[0]

        # retrieve correct model ids
        if task == "translation":
            # special case for translation pipeline which has multiple languages
            model_ids = []
            revisions = []
            tasks = []
            for translation_pair in task_dict["default"].keys():
                model_id, revision = task_dict["default"][translation_pair]["model"][framework]

                model_ids.append(model_id)
                revisions.append(revision)
                tasks.append(task + f"_{'_to_'.join(translation_pair)}")
        else:
            # normal case - non-translation pipeline
            model_id, revision = task_dict["default"]["model"][framework]

            model_ids = [model_id]
            revisions = [revision]
            tasks = [task]

        # check for equality
        for model_id, revision, task in zip(model_ids, revisions, tasks):
            # load default model
            try:
                set_seed_fn()
                model = auto_model_cls.from_pretrained(model_id)
            except ValueError:
                # first auto class is possible not compatible with model, go to next model class
                auto_model_cls = relevant_auto_classes[1]
                set_seed_fn()
                model = auto_model_cls.from_pretrained(model_id)

            # load default pipeline
            set_seed_fn()
            default_pipeline = pipeline(task)

            # compare pipeline model with default model
            models_are_equal = check_models_equal_fn(default_pipeline.model, model)
            self.assertTrue(models_are_equal, f"{task} model doesn't match pipeline.")

            logger.debug(f"{task} in {framework} succeeded with {model_id}.")

    def check_models_equal_pt(self, model1, model2):
        models_are_equal = True
        for model1_p, model2_p in zip(model1.parameters(), model2.parameters()):
            if model1_p.data.ne(model2_p.data).sum() > 0:
                models_are_equal = False

        return models_are_equal

    def check_models_equal_tf(self, model1, model2):
        models_are_equal = True
        for model1_p, model2_p in zip(model1.weights, model2.weights):
            if np.abs(model1_p.numpy() - model2_p.numpy()).sum() > 1e-5:
                models_are_equal = False

        return models_are_equal


class CustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, maybe_arg=2):
        input_ids = self.tokenizer(text, return_tensors="pt")
        return input_ids

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs["logits"].softmax(-1).numpy()


@is_pipeline_test
class CustomPipelineTest(unittest.TestCase):
    def test_warning_logs(self):
        transformers_logging.set_verbosity_debug()
        logger_ = transformers_logging.get_logger("mindnlp.transformers.pipelines.base")

        alias = "text-classification"
        # Get the original task, so we can restore it at the end.
        # (otherwise the subsequential tests in `TextClassificationPipelineTests` will fail)
        _, original_task, _ = PIPELINE_REGISTRY.check_task(alias)

        try:
            with CaptureLogger(logger_) as cm:
                PIPELINE_REGISTRY.register_pipeline(alias, PairClassificationPipeline)
            self.assertIn(f"{alias} is already registered", cm.out)
        finally:
            # restore
            PIPELINE_REGISTRY.supported_tasks[alias] = original_task

    def test_register_pipeline(self):
        PIPELINE_REGISTRY.register_pipeline(
            "custom-text-classification",
            pipeline_class=PairClassificationPipeline,
            model=AutoModelForSequenceClassification if is_mindspore_available() else None,
            default={"ms": "hf-internal-testing/tiny-random-distilbert"},
            type="text",
        )
        assert "custom-text-classification" in PIPELINE_REGISTRY.get_supported_tasks()

        _, task_def, _ = PIPELINE_REGISTRY.check_task("custom-text-classification")
        self.assertEqual(task_def["ms"], (AutoModelForSequenceClassification,) if is_mindspore_available() else ())
        self.assertEqual(task_def["type"], "text")
        self.assertEqual(task_def["impl"], PairClassificationPipeline)
        self.assertEqual(task_def["default"], {"model": {"ms": "hf-internal-testing/tiny-random-distilbert"}})

        # Clean registry for next tests.
        del PIPELINE_REGISTRY.supported_tasks["custom-text-classification"]

    # def test_dynamic_pipeline(self):
    #     PIPELINE_REGISTRY.register_pipeline(
    #         "pair-classification",
    #         pipeline_class=PairClassificationPipeline,
    #         model=AutoModelForSequenceClassification if is_mindspore_available() else None,
    #     )

    #     classifier = pipeline("pair-classification", model="hf-internal-testing/tiny-random-bert")

    #     # Clean registry as we won't need the pipeline to be in it for the rest to work.
    #     del PIPELINE_REGISTRY.supported_tasks["pair-classification"]

    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         classifier.save_pretrained(tmp_dir)
    #         # checks
    #         self.assertDictEqual(
    #             classifier.model.config.custom_pipelines,
    #             {
    #                 "pair-classification": {
    #                     "impl": "custom_pipeline.PairClassificationPipeline",
    #                     "ms": ("AutoModelForSequenceClassification",) if is_mindspore_available() else (),
    #                 }
    #             },
    #         )
    #         # Fails if the user forget to pass along `trust_remote_code=True`
    #         with self.assertRaises(ValueError):
    #             _ = pipeline(model=tmp_dir)

    #         new_classifier = pipeline(model=tmp_dir)
    #         # Using trust_remote_code=False forces the traditional pipeline tag
    #         old_classifier = pipeline("text-classification", model=tmp_dir)
    #     # Can't make an isinstance check because the new_classifier is from the PairClassificationPipeline class of a
    #     # dynamic module
    #     self.assertEqual(new_classifier.__class__.__name__, "PairClassificationPipeline")
    #     self.assertEqual(new_classifier.task, "pair-classification")
    #     results = new_classifier("I hate you", second_text="I love you")
    #     self.assertDictEqual(
    #         nested_simplify(results),
    #         {"label": "LABEL_0", "score": 0.505, "logits": [-0.003, -0.024]},
    #     )

    #     self.assertEqual(old_classifier.__class__.__name__, "TextClassificationPipeline")
    #     self.assertEqual(old_classifier.task, "text-classification")
    #     results = old_classifier("I hate you", text_pair="I love you")
    #     self.assertListEqual(
    #         nested_simplify(results),
    #         [{"label": "LABEL_0", "score": 0.505}],
    #     )

    # @require_mindspore_or_tf
    def test_cached_pipeline_has_minimum_calls_to_head(self):
        # Make sure we have cached the pipeline.
        _ = pipeline("text-classification", model="hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = pipeline("text-classification", model="hf-internal-testing/tiny-random-bert")
        # self.assertEqual(counter["GET"], 0)
        # self.assertEqual(counter["HEAD"], 1)
        # self.assertEqual(counter.total_calls, 1)

    # @require_mindspore
    # def test_chunk_pipeline_batching_single_file(self):
    #     # Make sure we have cached the pipeline.
    #     pipe = pipeline(model="hf-internal-testing/tiny-random-Wav2Vec2ForCTC")
    #     ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
    #     audio = ds[40]["audio"]["array"]

    #     pipe = pipeline(model="hf-internal-testing/tiny-random-Wav2Vec2ForCTC")
    #     # For some reason scoping doesn't work if not using `self.`
    #     self.COUNT = 0
    #     forward = pipe.model.forward

    #     def new_forward(*args, **kwargs):
    #         self.COUNT += 1
    #         return forward(*args, **kwargs)

    #     pipe.model.forward = new_forward

    #     for out in pipe(audio, return_timestamps="char", chunk_length_s=3, stride_length_s=[1, 1], batch_size=1024):
    #         pass

    #     self.assertEqual(self.COUNT, 1)
