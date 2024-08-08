# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""MindNLP gpt_bigcode config"""


from ...configuration_utils import PretrainedConfig

GPT_BIGCODE_SUPPORT_LIST = ["gpt_bigcode-santacoder"]


__all__ = ['GPTBigCodeConfig']


class GPTBigCodeConfig(PretrainedConfig):
    r"""
    GPT BigCode config
    """
    model_type = "gpt_bigcode"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_approximate",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        attention_softmax_in_fp32=True,
        scale_attention_softmax_in_fp32=True,
        multi_query=True,
        **kwargs,
    ):
        """
        __init__
        
        Initialize a new GPTBigCodeConfig object.
        
        Args:
            vocab_size (int, optional): The size of the vocabulary. Default is 50257.
            n_positions (int, optional): The maximum sequence length for the model. Default is 1024.
            n_embd (int, optional): The dimension of the embeddings and hidden states. Default is 768.
            n_layer (int, optional): The number of layers in the model. Default is 12.
            n_head (int, optional): The number of attention heads in the model. Default is 12.
            n_inner (int, optional): The inner dimension of the feedforward layers. Default is None.
            activation_function (str, optional): The activation function used in the model. Default is 'gelu_approximate'.
            resid_pdrop (float, optional): The dropout probability for residual connections. Default is 0.1.
            embd_pdrop (float, optional): The dropout probability for embeddings. Default is 0.1.
            attn_pdrop (float, optional): The dropout probability for attention layers. Default is 0.1.
            layer_norm_epsilon (float, optional): The epsilon value for layer normalization. Default is 1e-05.
            initializer_range (float, optional): The range for parameter initializers. Default is 0.02.
            scale_attn_weights (bool, optional): Whether to scale the attention weights. Default is True.
            use_cache (bool, optional): Whether to use caching during inference. Default is True.
            bos_token_id (int, optional): The token id for the beginning of sequence. Default is 50256.
            eos_token_id (int, optional): The token id for the end of sequence. Default is 50256.
            attention_softmax_in_fp32 (bool, optional): Whether to use fp32 for attention softmax. Default is True.
            scale_attention_softmax_in_fp32 (bool, optional): Whether to scale attention softmax in fp32. Default is True.
            multi_query (bool, optional): Whether to use multi-query attention. Default is True.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32
        self.multi_query = multi_query

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
