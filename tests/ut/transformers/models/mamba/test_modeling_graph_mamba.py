# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import math
import unittest
from typing import Dict, List, Tuple
from unittest.util import safe_repr

from parameterized import parameterized

import numpy as np
from mindnlp.transformers import AutoTokenizer, MambaConfig
from mindnlp.utils.testing_utils import require_mindspore, slow, is_mindspore_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
# from ...test_pipeline_mixin import PipelineTesterMixin


if is_mindspore_available():
    import mindspore
    from mindspore import ops

    from mindnlp.transformers import (
        MSMambaForCausalLM as MambaForCausalLM,
        MSMambaModel as MambaModel,
    )


@require_mindspore
class MambaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.model_id = "state-spaces/mamba-2.8b-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @slow
    def test_simple_generate(self):
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
        tokenizer.pad_token = tokenizer.eos_token

        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", ms_dtype=mindspore.float16)
        model.config.use_cache = True
        input_ids = tokenizer("Hey how are you doing?", return_tensors="ms")["input_ids"]

        out = model.generate(input_ids, do_sample=False, max_new_tokens=10)
        output_sentence = tokenizer.decode(out[0, :])
        self.assertEqual(output_sentence, "Hey how are you doing?\n\nI'm so glad you're here.")

        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", ms_dtype=mindspore.float16)
        logits = model(input_ids=input_ids).logits

        EXPECTED_LOGITS_NO_GRAD = mindspore.tensor(
            [
                -55.6875, -69.8750, -49.9062, -51.7500, -57.6875, -57.9375, -56.9688,
                -57.9375, -54.6875, -55.9375, -55.3125, -58.0938, -60.5625, -47.0000,
                -52.0312, -49.7812, -55.9375, -57.9062, -56.7812, -57.1250, -57.3438,
                -58.3125, -57.8125, -58.7812, -59.6250, -59.0938, -58.7188, -52.9375,
                -53.4688, -57.3750, -56.9375, -55.7500, -53.3125, -55.8438, -57.0000,
                -56.9062, -56.2188, -54.7188, -56.4375, -57.5000
            ], dtype=mindspore.float32)  # fmt: skip

        self.assertTrue(np.allclose(logits[0, 0, :40].asnumpy(), EXPECTED_LOGITS_NO_GRAD.asnumpy(), rtol=1e-2, atol=1e-2))

    @slow
    def test_simple_generate_cuda_kernels_tiny(self):
        expected_output = "Hello my name is John and I am a newbie to the world"

        input_ids = self.tokenizer("Hello my name is", return_tensors="ms").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", ms_dtype=mindspore.float16)

        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @slow
    def test_simple_generate_cuda_kernels_small(self):
        expected_output = "Hello my name is\n\nI am a\n\nI am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="ms").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-790m-hf", ms_dtype=mindspore.float16)

        output = model.generate(input_ids, max_new_tokens=10)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @slow
    def test_simple_generate_cuda_kernels_mid(self):
        expected_output = "Hello my name is John and I am a\n\nI am a single father of a beautiful daughter. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="ms").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf", ms_dtype=mindspore.float16)

        output = model.generate(input_ids, max_new_tokens=20)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)

    @slow
    def test_simple_generate_cuda_kernels_big(self):
        expected_output = "Hello my name is John and I am a new member of this forum. I am a retired Marine and I am a member of the Marine Corps League. I am a"

        input_ids = self.tokenizer("Hello my name is", return_tensors="ms").input_ids
        model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf", ms_dtype=mindspore.float16)

        output = model.generate(input_ids, max_new_tokens=30)
        output_sentence = self.tokenizer.decode(output[0].tolist())

        self.assertEqual(output_sentence, expected_output)
