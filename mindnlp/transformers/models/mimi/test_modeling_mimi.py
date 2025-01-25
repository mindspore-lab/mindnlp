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

import unittest

import numpy as np
import mindspore as ms
from mindspore import Tensor, context

from transformers import MimiConfig
from transformers.testing_utils import require_mindspore, slow
from transformers.models.mimi.modeling_mimi import (
    MimiModel,
    MimiEncoder,
    MimiDecoder,
    MimiTransformerModel,
    MimiResidualVectorQuantizer,
)

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def create_and_check_model_forward(
    config,
    input_values,
    padding_mask,
    num_quantizers,
):
    model = MimiModel(config)
    model.set_train(False)

    result = model(
        input_values,
        padding_mask=padding_mask,
        num_quantizers=num_quantizers,
    )

    self.parent.assertEqual(
        result.audio_codes.shape,
        (input_values.shape[0], num_quantizers, input_values.shape[-1] // 320),
    )

    self.parent.assertEqual(
        result.audio_values.shape,
        input_values.shape,
    )

class MimiModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = MimiModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MimiConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    @require_mindspore
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_forward(*config_and_inputs)

    @slow
    @require_mindspore
    def test_model_from_pretrained(self):
        model = MimiModel.from_pretrained("kyutai/mimi")
        self.assertIsNotNone(model)

class MimiModelTester:
    def __init__(self, parent, batch_size=2):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = 1
        self.sequence_length = 32000  # ~1.3 seconds at 24kHz
        self.is_training = False
        self.use_labels = False
        self.hidden_size = 128
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 512
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 8000
        self.num_quantizers = 8
        self.type_vocab_size = 2
        self.initializer_range = 0.02
        self.layer_norm_eps = 1e-12
        self.pad_token_id = 0
        self.position_embedding_type = "absolute"
        self.use_cache = True
        self.vocab_size = 50265
        self.num_filters = 32
        self.codebook_size = 1024

    def prepare_config_and_inputs(self):
        input_values = Tensor(
            np.random.randn(self.batch_size, self.num_channels, self.sequence_length),
            dtype=ms.float32
        )
        padding_mask = Tensor(
            np.ones((self.batch_size, self.num_channels, self.sequence_length)),
            dtype=ms.float32
        )

        config = self.get_config()

        return config, input_values, padding_mask, self.num_quantizers

    def get_config(self):
        return MimiConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            num_quantizers=self.num_quantizers,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            position_embedding_type=self.position_embedding_type,
            use_cache=self.use_cache,
            vocab_size=self.vocab_size,
            num_filters=self.num_filters,
            codebook_size=self.codebook_size,
        )

    def create_and_check_model_forward(
        self,
        config,
        input_values,
        padding_mask,
        num_quantizers,
    ):
        model = MimiModel(config)
        model.set_train(False)

        result = model(
            input_values,
            padding_mask=padding_mask,
            num_quantizers=num_quantizers,
        )

        self.parent.assertEqual(
            result.audio_codes.shape,
            (input_values.shape[0], num_quantizers, input_values.shape[-1] // 320),
        )

        self.parent.assertEqual(
            result.audio_values.shape,
            input_values.shape,
        )

class ConfigTester:
    def __init__(self, parent, config_class=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.inputs_dict = kwargs

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_size"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        self.parent.assertTrue(hasattr(config, "num_hidden_layers"))

    def run_common_tests(self):
        self.create_and_test_config_common_properties()

if __name__ == "__main__":
    unittest.main() 