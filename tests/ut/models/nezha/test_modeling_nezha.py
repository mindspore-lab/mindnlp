# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Test Nezha
"""
import unittest
import random
import numpy as np

import mindspore
from mindspore import Tensor

from mindnlp.models.nezha import (NezhaConfig,
                                  NezhaRelativePositionsEncoding,
                                  NezhaEmbeddings,
                                  NezhaSelfAttention,
                                  NezhaSelfOutput,
                                  NezhaAttention,
                                  NezhaIntermediate,
                                  NezhaOutput,
                                  NezhaLayer,
                                  NezhaEncoder,
                                  NezhaPooler,
                                  NezhaPredictionHeadTransform,
                                  NezhaLMPredictionHead,
                                  NezhaOnlyMLMHead,
                                  NezhaOnlyNSPHead,
                                  NezhaPreTrainingHeads,
                                  NezhaModel,
                                  NezhaForPreTraining,
                                  NezhaForMaskedLM,
                                  NezhaForNextSentencePrediction,
                                  NezhaForSequenceClassification,
                                  NezhaForMultipleChoice,
                                  NezhaForTokenClassification,
                                  NezhaForQuestionAnswering)
class TestNezhaBasicModule(unittest.TestCase):
    r"""
    Test Nezha Basic Module
    """
    def setUp(self):
        self.inputs = None

    def test_nezha_relative_positions_encoding(self):
        r"""
        Test NezhaRelativePositionsEncoding
        """
        length = 10
        depth = 20
        model = NezhaRelativePositionsEncoding(length, depth)
        inputs = random.randint(0, 10)
        outputs = model(inputs)
        assert outputs.shape == (inputs, inputs, 20)

    def test_nezha_embeddings(self):
        r"""
        Test NezhaEmbeddings
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaEmbeddings(config)
        inputs = Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_self_attention(self):
        r"""
        Test NezhaSelfAttention
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaSelfAttention(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_self_output(self):
        r"""
        Test NezhaSelfOutput
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaSelfOutput(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs, inputs)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_attention(self):
        r"""
        Test NezhaAttention
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaAttention(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_intermediate(self):
        r"""
        Test NezhaIntermediate
        """
        config = NezhaConfig(hidden_size=48, intermediate_size=192)
        model = NezhaIntermediate(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 192)

    def test_nezha_output(self):
        r"""
        Test NezhaOutput
        """
        config = NezhaConfig(hidden_size=48, intermediate_size=192)
        model = NezhaOutput(config)
        inputs1 = Tensor(np.random.randn(4, 16, 192), mindspore.float32)
        inputs2 = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs1, inputs2)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_layer(self):
        r"""
        Test NezhaLayer
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaLayer(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_encoder(self):
        r"""
        Test NezhaEncoder
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaEncoder(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)

    def test_nezha_pooler(self):
        r"""
        Test NezhaPooler
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaPooler(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 48)

    def test_nezha_prediction_head_transform(self):
        r"""
        Test NezhaPredictionHeadTransform
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaPredictionHeadTransform(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 48)

    def test_nezha_lm_prediction_head(self):
        r"""
        Test NezhaLMPredictionHead
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaLMPredictionHead(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 64)

    def test_nezha_only_mlm_head(self):
        r"""
        Test NezhaOnlyMLMHead
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaOnlyMLMHead(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 64)

    def test_nezha_only_nsp_head(self):
        r"""
        Test NezhaOnlyNSPHead
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaOnlyNSPHead(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs)
        assert outputs.shape == (4, 16, 2)

    def test_nezha_pretraining_heads(self):
        r"""
        Test NezhaPreTrainingHeads
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaPreTrainingHeads(config)
        inputs = Tensor(np.random.randn(4, 16, 48), mindspore.float32)
        outputs = model(inputs, inputs)
        assert outputs[0].shape == (4, 16, 64)
        assert outputs[1].shape == (4, 16, 2)


class TestModelingNezha(unittest.TestCase):
    r"""
    Test Nezha Model
    """
    def setUp(self):
        self.inputs = None

    def test_nezha_model(self):
        r"""
        Test NezhaModel
        """
        config = NezhaConfig(hidden_size=48)
        model = NezhaModel(config)
        inputs = Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 48)
        assert outputs[1].shape == (4, 48)

    def test_nezha_for_pretraining(self):
        r"""
        Test NezhaForPreTraining
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForPreTraining(config)
        inputs= Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 64)
        assert outputs[1].shape == (4, 2)

    def test_nezha_for_masked_lm(self):
        r"""
        Test NezhaForMaskedLM
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForMaskedLM(config)
        inputs= Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 64)

    def test_nezha_for_next_sentence_prediction(self):
        r"""
        Test NezhaForNextSentencePrediction
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForNextSentencePrediction(config)
        inputs= Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 2)

    def test_nezha_for_sequence_classification(self):
        r"""
        Test NezhaForSequenceClassification
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForSequenceClassification(config)
        inputs= Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 2)

    def test_nezha_for_multiple_choice(self):
        r"""
        Test NezhaForMultipleChoice
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForMultipleChoice(config)
        inputs= Tensor(np.random.randn(4, 4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 4)

    def test_nezha_for_token_classification(self):
        r"""
        Test NezhaforTokenClassification
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForTokenClassification(config)
        inputs= Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16, 2)

    def test_for_question_answering(self):
        r"""
        Test NezhaForQuestionAnswering
        """
        config = NezhaConfig(hidden_size=48, vocab_size=64)
        model = NezhaForQuestionAnswering(config)
        inputs = Tensor(np.random.randn(4, 16), mindspore.int64)
        outputs = model(inputs)
        assert outputs[0].shape == (4, 16)
        assert outputs[1].shape == (4, 16)
