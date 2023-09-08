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
import mindspore
import numpy as np
from mindspore import Tensor

import mindnlp
from mindnlp.models.codegen import codegen_config, codegen
from ..model_test import ModelTest

class TestModelingCodeGen(ModelTest):
    r"""
    Test CodeGen
    """

    def setUp(self):
        """
        Set up.
        """
        super().setUp()
        self.config = codegen_config.CodeGenConfig(vocab_size=1000,
                                                   hidden_size=128,
                                                   n_layer=2,
                                                   n_head=8,
                                                   )

    def test_codegen_attention(self):
        r"""
        Test CodeGen Attention
        """
        model = codegen.CodeGenAttention(self.config)
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 2, 512)), mindspore.float32)

        attn_output, _ = model(hidden_states)
        assert attn_output.shape == (2, 2, 512)

    def test_codegen_mlp(self):
        r"""
        Test CodeGen MLP
        """
        intermediate_size = 100
        model = codegen.CodeGenMLP(intermediate_size, self.config)
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 2, 512)), mindspore.float32)

        hidden_states = model(hidden_states)
        assert hidden_states.shape == (2, 2, 512)

    def test_codegen_block(self):
        r"""
            Test CodeGen BLOCK
        """
        model = codegen.CodeGenBlock(self.config)
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        hidden_states = Tensor(np.random.randint(0, 10, (2, 2, 512)), mindspore.float32)

        hidden_states = model(hidden_states)
        assert hidden_states[0].shape == (2, 2, 512)

    def test_codegen_model(self):
        r"""
            Test CodeGen MODEL
        """
        model = codegen.CodeGenModel(self.config)
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(0, 10, (2, 2, 512)), mindspore.int32)

        input_ids = model(input_ids)
        assert input_ids[0].shape == (2, 2, 512, 512)

    def test_codegen_forcausallm(self):
        r"""
            Test CodeGen FORCAUSALLM
        """
        model = codegen.CodeGenForCausalLM(self.config)
        if self.use_amp:
            model = mindnlp._legacy.amp.auto_mixed_precision(model)

        input_ids = Tensor(np.random.randint(0, 10, (2, 2, 512)), mindspore.int32)

        input_ids = model(input_ids)
        assert input_ids[0].shape == (2, 2, 512, 1000)
