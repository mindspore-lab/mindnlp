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
Test XLM
"""
import unittest
import numpy as np
import mindspore
from mindnlp.models.xlm import xlm_config
from mindnlp.models import xlm
class TestXlm(unittest.TestCase):
    """
    Test XLM Models
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up config
        """
        cls.input = None


    def test_xlm_predlayer(self):
        """
        Test xlm_XLMPredLayer
        """
        xlmconfig = xlm_config.XLMConfig(n_words=22)
        xlm_predlayer = xlm.XLMPredLayer(xlmconfig)
        input_ids = mindspore.Tensor(np.random.randint(0, 1000, (2, 2048)),mindspore.float32)
        output = xlm_predlayer(input_ids)
        assert output[0].shape == (2,22)

    def test_xlm_multiheadattention(self):
        """
        test xlm_multiheadattention
        """
        xlmconfig = xlm_config.XLMConfig(n_words=22)
        xlm_multiheadattention = xlm.MultiHeadAttention(n_heads=xlmconfig.n_heads,dim = xlmconfig.emb_dim,config=xlmconfig)
        input_ids = mindspore.Tensor(np.random.randint(0, 1000, (2, 3,2048)),mindspore.float32)
        mask = mindspore.Tensor(np.random.randint(0, 1000, (2, 3,1,1)),mindspore.float32)
        output = xlm_multiheadattention(input = input_ids,mask = mask)
        assert output[0].shape == (2,3,2048)

    def  test_xlm_transformerffn(self):
        """
        test xlm_TransformerFFN
        """
        xlmconfig = xlm_config.XLMConfig(n_words=22)
        xlm_transformerffn = xlm.TransformerFFN(xlmconfig.emb_dim,xlmconfig.emb_dim*4, xlmconfig.emb_dim,xlmconfig)
        input_ids = mindspore.Tensor(np.random.randint(0, 1000, (2, 2048)),mindspore.float32)
        output = xlm_transformerffn(input_ids)
        assert output.shape==(2,2048)


    def  test_xlm_xlmmodel(self):
        """
        test xlm_TransformerFFN
        """
        xlmconfig = xlm_config.XLMConfig(n_words=22)
        xlm_xlmmodel = xlm.XLMModel(xlmconfig)
        input_ids = mindspore.Tensor(np.random.randint(0, 1000, (1, 512)),mindspore.int32)
        output = xlm_xlmmodel(input_ids)
        assert output[0].shape==(1,512,2048)
