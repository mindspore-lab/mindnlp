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
"""Test CodeGen"""
import unittest

import mindspore
import numpy as np
from mindspore import Tensor

from mindnlp.models.codegen import codegen_config, codegen


class TestModelingCodeGen(unittest.TestCase):
    r"""
    Test CodeGen
    """

    def setUp(self):
        """
        Set up.
        """
        self.input = None

    def test_codegen_attention(self):
        r"""
        Test CodeGen Attention
        """
        config = codegen_config.CodeGenConfig()
        model = codegen.CodeGenAttention(config)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 2, 4096)), mindspore.float32)

        attn_output, _ = model(hidden_states)
        assert attn_output.shape == (2, 2, 4096)
