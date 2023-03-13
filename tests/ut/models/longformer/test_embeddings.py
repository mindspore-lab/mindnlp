# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Test Longformer"""
import unittest
import numpy as np
from mindspore import Tensor
from mindnlp.models.longformer.longformer import LongformerEmbeddings
from mindnlp.models.longformer.longformer_config import LongformerConfig
class TestModelingBert(unittest.TestCase):
    r"""
    Test model bert
    """
    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_modeling_longformer_embedding(self):
        r"""
        Test model bert with pynative mode
        """
        ms_config = LongformerConfig()
        ms_model = LongformerEmbeddings(ms_config)
        ms_model.set_train(False)
        tensor = np.random.randint(1, 10, (2, 2))
        ms_input_ids = Tensor.from_numpy(tensor)
        ms_outputs = ms_model.forward(ms_input_ids)
        assert (2, 2, 768) == ms_outputs.shape
