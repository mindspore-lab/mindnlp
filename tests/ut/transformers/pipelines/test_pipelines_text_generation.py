
import unittest

from mindnlp.transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    TextGenerationPipeline,
    pipeline,
)
from mindnlp.utils import logging
from mindnlp.utils.testing_utils import is_pipeline_test, nested_simplify, require_mindspore, slow, CaptureLogger
from .test_pipelines_common import ANY


@is_pipeline_test
class TextGenerationPipelineTests(unittest.TestCase):
    model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING

    @require_mindspore
    def test_small_model(self):
        text_generator = pipeline(task="text-generation", model="t5-small")
        # Using `do_sample=False` to force deterministic output
        outputs = text_generator("This is a test", do_sample=False)
        self.assertEqual(
            outputs,
            [
                {
                    "generated_text": (
                        "This is a test is a test test This is a test test This is "
                        "a test"
                    )
                }
            ],
        )

        outputs = text_generator(["This is a test", "This is a second test"])
        self.assertEqual(
            outputs,
            [
                [
                    {
                        "generated_text": (
                        "This is a test is a test test This is a test test This is "
                        "a test")
                    }
                ],
                [
                    {
                        "generated_text": (
                            "This is a second test is a second test test this is a "
                            "second test test this is")
                    }
                ],
            ],
        )

        outputs = text_generator("This is a test", do_sample=True, num_return_sequences=2, return_tensors=True)
        self.assertEqual(
            outputs,
            [
                {"generated_token_ids": ANY(list)},
                {"generated_token_ids": ANY(list)},
            ],
        )

        ## -- test tokenizer_kwargs
        test_str = "testing tokenizer kwargs. using truncation must result in a different generation."
        input_len = len(text_generator.tokenizer(test_str)["input_ids"])
        output_str, output_str_with_truncation = (
            text_generator(test_str, do_sample=False, return_full_text=False, min_new_tokens=1)[0]["generated_text"],
            text_generator(
                test_str,
                do_sample=False,
                return_full_text=False,
                min_new_tokens=1,
                truncation=True,
                max_length=input_len + 1,
            )[0]["generated_text"],
        )
        assert output_str != output_str_with_truncation  # results must be different because one had truncation

        # -- what is the point of this test? padding is hardcoded False in the pipeline anyway
        text_generator.tokenizer.pad_token_id = text_generator.model.config.eos_token_id
        text_generator.tokenizer.pad_token = "<pad>"
        outputs = text_generator(
            ["This is a test", "This is a second test"],
            do_sample=True,
            num_return_sequences=2,
            batch_size=2,
            return_tensors=True,
        )
        self.assertEqual(
            outputs,
            [
                [
                    {"generated_token_ids": ANY(list)},
                    {"generated_token_ids": ANY(list)},
                ],
                [
                    {"generated_token_ids": ANY(list)},
                    {"generated_token_ids": ANY(list)},
                ],
            ],
        )

    @require_mindspore
    def test_small_chat_model(self):
        text_generator = pipeline(
            task="text-generation", model="gpt2"
        )
        # Using `do_sample=False` to force deterministic output
        chat1 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a test"},
            {"role": "assistant", "content": "This is a reply"},
        ]
        chat2 = [
            {"role": "system", "content": "This is a system message."},
            {"role": "user", "content": "This is a second test"},
            {"role": "assistant", "content": "This is a reply"},
        ]
        outputs = text_generator(chat1, do_sample=False, max_new_tokens=10)
        expected_chat1 = chat1 + [
            {
                "role": "assistant",
                "content": "The following is a list of the most popular and",
            }
        ]
        self.assertEqual(
            outputs,
            [
                {"generated_text": expected_chat1},
            ],
        )

        outputs = text_generator([chat1, chat2], do_sample=False, max_new_tokens=10)
        expected_chat2 = chat2 + [
            {
                "role": "assistant",
                "content": "The following is a list of the most popular and",
            }
        ]

        self.assertEqual(
            outputs,
            [
                [{"generated_text": expected_chat1}],
                [{"generated_text": expected_chat2}],
            ],
        )


    def get_test_pipeline(self, model, tokenizer, processor):
        text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        return text_generator, ["HuggingFace is in", "This is another test"]

    def test_stop_sequence_stopping_criteria(self):
        prompt = """Hello I believe in"""
        text_generator = pipeline("text-generation", model="t5-small")
        output = text_generator(prompt)
        self.assertEqual(
            output,
            [{"generated_text":
                    "Hello I believe inHello Hello Hello I believe in believe in in believe in in Hello Hello Hello Hello",
            }]
        )

        output = text_generator(prompt, stop_sequence=" in")
        self.assertEqual(output, [{"generated_text": "Hello I believe inHello Hello Hello I believe in"}])

    def run_pipeline_test(self, text_generator, _):
        model = text_generator.model
        tokenizer = text_generator.tokenizer

        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator("This is a test", return_full_text=False)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        outputs = text_generator("This is a test")
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertNotIn("This is a test", outputs[0]["generated_text"])

        outputs = text_generator("This is a test", return_full_text=True)
        self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        self.assertTrue(outputs[0]["generated_text"].startswith("This is a test"))

        outputs = text_generator(["This is great !", "Something else"], num_return_sequences=2, do_sample=True)
        self.assertEqual(
            outputs,
            [
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
            ],
        )

        if text_generator.tokenizer.pad_token is not None:
            outputs = text_generator(
                ["This is great !", "Something else"], num_return_sequences=2, batch_size=2, do_sample=True
            )
            self.assertEqual(
                outputs,
                [
                    [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                    [{"generated_text": ANY(str)}, {"generated_text": ANY(str)}],
                ],
            )

        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_full_text=True, return_text=True)
        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_full_text=True, return_tensors=True)
        with self.assertRaises(ValueError):
            outputs = text_generator("test", return_text=True, return_tensors=True)

        # Empty prompt is slighly special
        # it requires BOS token to exist.
        # Special case for Pegasus which will always append EOS so will
        # work even without BOS.
        if (
            text_generator.tokenizer.bos_token_id is not None
            or "Pegasus" in tokenizer.__class__.__name__
            or "Git" in model.__class__.__name__
        ):
            outputs = text_generator("")
            self.assertEqual(outputs, [{"generated_text": ANY(str)}])
        else:
            with self.assertRaises((ValueError, AssertionError)):
                outputs = text_generator("")

        if text_generator.framework == "tf":
            # TF generation does not support max_new_tokens, and it's impossible
            # to control long generation with only max_length without
            # fancy calculation, dismissing tests for now.
            return
        # We don't care about infinite range models.
        # They already work.
        # Skip this test for XGLM, since it uses sinusoidal positional embeddings which are resized on-the-fly.
        EXTRA_MODELS_CAN_HANDLE_LONG_INPUTS = [
            "RwkvForCausalLM",
            "XGLMForCausalLM",
            "GPTNeoXForCausalLM",
            "FuyuForCausalLM",
        ]
        if (
            tokenizer.model_max_length < 10000
            and text_generator.model.__class__.__name__ not in EXTRA_MODELS_CAN_HANDLE_LONG_INPUTS
        ):
            # Handling of large generations
            with self.assertRaises((RuntimeError, IndexError, ValueError, AssertionError)):
                text_generator("This is a test" * 500, max_new_tokens=20)

            outputs = text_generator("This is a test" * 500, handle_long_generation="hole", max_new_tokens=20)
            # Hole strategy cannot work
            with self.assertRaises(ValueError):
                text_generator(
                    "This is a test" * 500,
                    handle_long_generation="hole",
                    max_new_tokens=tokenizer.model_max_length + 10,
                )

    @require_mindspore
    def test_small_model_pt_bloom_accelerate(self):
        import mindspore
        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom",
            model_kwargs={"ms_dtype": mindspore.float16},
        )
        self.assertEqual(pipe.model.lm_head.weight.dtype, mindspore.float16)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

        # Upgraded those two to real pipeline arguments (they just get sent for the model as they're unlikely to mean anything else.)
        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom", ms_dtype=mindspore.float16)
        self.assertEqual(pipe.model.lm_head.weight.dtype, mindspore.float16)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

        pipe = pipeline(model="hf-internal-testing/tiny-random-bloom",ms_dtype=mindspore.float32)
        self.assertEqual(pipe.model.lm_head.weight.dtype, mindspore.float32)
        out = pipe("This is a test")
        self.assertEqual(
            out,
            [
                {
                    "generated_text": (
                        "This is a test test test test test test test test test test test test test test test test"
                        " test"
                    )
                }
            ],
        )

    @require_mindspore
    def test_small_model_fp16(self):
        import mindspore

        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom",
            ms_dtype=mindspore.float16,
        )
        pipe("This is a test")

    @require_mindspore
    def test_pipeline_accelerate_top_p(self):
        import mindspore

        pipe = pipeline(
            model="hf-internal-testing/tiny-random-bloom", ms_dtype=mindspore.float16
        )
        pipe("This is a test", do_sample=True, top_p=0.5)

    def test_pipeline_length_setting_warning(self):
        prompt = """Hello world"""
        text_generator = pipeline("text-generation", model="hf-internal-testing/tiny-random-gpt2")
        logger = logging.get_logger("mindnlp.transformers.generation.utils")
        logger_msg = "Both `max_new_tokens`"  # The beggining of the message to be checked in this test

        # Both are set by the user -> log warning
        with CaptureLogger(logger) as cl:
            _ = text_generator(prompt, max_length=10, max_new_tokens=1)
        self.assertIn(logger_msg, cl.out)

        # The user only sets one -> no warning
        with CaptureLogger(logger) as cl:
            _ = text_generator(prompt, max_new_tokens=1)
        self.assertNotIn(logger_msg, cl.out)

        with CaptureLogger(logger) as cl:
            _ = text_generator(prompt, max_length=10)
        self.assertNotIn(logger_msg, cl.out)