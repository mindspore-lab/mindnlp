# Copyright 2024 The Qwen team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Qwen3 model."""

import gc
import unittest
from mindnlp.utils.testing_utils import (
    slow,
)


import mindspore
from mindnlp.core import ops, no_grad

from mindnlp.transformers import (
    Qwen3ForCausalLM,
    AutoTokenizer,
)



class Qwen3IntegrationTest(unittest.TestCase):

    @slow
    def test_model_600m_logits(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]
        model = Qwen3ForCausalLM.from_pretrained("/mnt/data/zqh/llm/Qwen3-0.6B")
        # model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        input_ids = mindspore.tensor([input_ids])
        with no_grad():
            out = model(input_ids).logits.float()
        # Expected mean on dim = -1
        EXPECTED_MEAN = mindspore.tensor([[-0.9166, -0.2534,  0.3173,  1.4262,  0.6352,  0.3361,  0.8639,  0.2072]])
        assert ops.allclose(out.mean(-1), EXPECTED_MEAN, rtol=1e-2, atol=1e-2)
        # slicing logits[0, 0, 0:30]
        EXPECTED_SLICE = mindspore.tensor([4.4758, 4.5917, 4.9683, 1.2506, 3.6566, 2.7168, 3.0117, 2.5091, 2.2319,
        6.0900, 3.3463, 2.0949, 5.6522, 1.6135, 3.5464, 3.2986, 4.0523, 3.8702,
        4.1618, 3.5288, 3.4191, 3.5210, 2.9291, 3.2516, 2.1061, 3.7249, 2.4170,
        3.8529, 1.1164, 4.1829])  # fmt: skip
        assert ops.allclose(out[0, 0, :30], EXPECTED_SLICE, rtol=1e-2, atol=1e-2)

        del model
        gc.collect()

    @slow
    def test_model_600m_generation(self):
        EXPECTED_TEXT_COMPLETION = """My favourite condiment is 100% natural, and I have a recipe that uses 100% natural cond"""
        prompt = "My favourite condiment is "
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", use_fast=False)
        model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        input_ids = tokenizer.encode(prompt, return_tensors="ms")