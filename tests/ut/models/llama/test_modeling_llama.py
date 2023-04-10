# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test Llama"""
import unittest
import numpy as np
import mindspore

from mindspore import Tensor
from mindnlp.models.llama import llama, llama_config

class TestModelingLlama(unittest.TestCase):
    """
    Test Llama
    """
    def setUp(self):
        """
        Set up.
        """
        self.config = llama_config.LlamaConfig()
        self.config.max_batch_size = 2
        self.config.dim = 128
        self.config.max_seq_len = 256
        self.config.vocab_size = 256 # define by tokenizer
        self.input = None

    def test_llama_rmsnorm(self):
        """
        test llama rmsnorm
        """
        config = self.config
        model = llama.RMSNorm(config.dim)

        rmsnorm_input = Tensor(np.random.randint(0, 10,
                                                 (config.max_batch_size, config.max_seq_len, config.dim)
                                                 ), mindspore.float32)

        output = model(rmsnorm_input)

        assert output.shape == (config.max_batch_size, config.max_seq_len, config.dim)

    def test_llama_attention(self):
        '''
        test llama attention
        '''
        config = self.config
        model = llama.Attention(config)
        attention_input = Tensor(np.random.randint(0, 10,
                                (config.max_batch_size, config.max_seq_len, config.dim))
                                , mindspore.float32)
        freqs_cis = llama.precompute_freqs_cis(config.dim // config.n_heads,
                                                config.max_seq_len * 2)[0:config.max_seq_len]
        output = model(attention_input, start_pos=0, freqs_cis=freqs_cis, mask=None)

        assert output.shape == (config.max_batch_size, config.max_seq_len, config.dim)

    def test_llama_feedforward(self):
        '''
        test llama test_llama_feedforward
        '''
        config = self.config
        model = llama.FeedForward(config.dim, config.dim * 4, config.multiple_of)
        feedforward_input = Tensor(np.random.randint(0, 10,
                                (config.max_batch_size, config.max_seq_len, config.dim))
                                , mindspore.float32)
        output = model(feedforward_input)

        assert output.shape == (config.max_batch_size, config.max_seq_len, config.dim)

    def test_llama_transformerblock(self):
        '''
        test_llama_transformerblock
        '''
        config = self.config
        model = llama.TransformerBlock(0, config)
        transformerblock_input = Tensor(np.random.randint(0, 10,
                                (config.max_batch_size, config.max_seq_len, config.dim))
                                , mindspore.float32)
        freqs_cis = llama.precompute_freqs_cis(config.dim // config.n_heads,
                                                config.max_seq_len * 2)[0:config.max_seq_len]
        output = model(_x=transformerblock_input, freqs_cis=freqs_cis, start_pos=0, mask=None)

        assert output.shape == (config.max_batch_size, config.max_seq_len, config.dim)

    def test_llama_transformer(self):
        '''
        test_llama_transformer
        '''
        config = self.config
        model = llama.Transformer(config)
        tokens = Tensor(np.random.randint(0, config.vocab_size,
                                (config.max_batch_size, config.max_seq_len))
                                , mindspore.int32)
        output = model(tokens, 0)

        assert output.shape == (config.max_batch_size, config.max_seq_len)
