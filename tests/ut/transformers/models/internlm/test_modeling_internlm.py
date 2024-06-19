"""Test InternLM"""
import gc
import os
import unittest

import mindspore
import numpy as np
from mindspore import Tensor
from mindnlp.transformers.models.internlm import InternLMConfig, InternLMModel, InternLMForCausalLM, \
    InternLMForSequenceClassification
from mindnlp.utils.testing_utils import slow
from tests.common import MindNLPTestCase

class TestModelingBaiChuan(MindNLPTestCase):
    r"""
    Test BaiChuan
    """

    def setUp(self):
        """
        Set up.
        """
        self.config_7b = InternLMConfig(vocab_size=1000, num_hidden_layers=2)

    @slow
    def test_7b_model(self):
        r"""
        Test Model
        """
        model = InternLMModel(self.config_7b)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 4096)


    @slow
    def test_internlm_for_causal_lm_7b(self):
        r"""
        Test InternLMForCausalLM
        """
        model = InternLMForCausalLM(self.config_7b)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 128, 1000)
    
    @slow
    def test_internlm_for_sequence_classification_7b(self):
        r"""
        Test InternLMForSequenceClassification
        """
        model = InternLMForSequenceClassification(self.config_7b)
        input_ids = Tensor(np.random.randint(0, 100, (1, 128)), mindspore.int32)
        outputs = model(input_ids=input_ids)
        assert outputs[0].shape == (1, 2)
