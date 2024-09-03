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
"""ChatGLM2 Config"""
from ...configuration_utils import PretrainedConfig

class ChatGLM2Config(PretrainedConfig):
    """ChatGLM2Config"""
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        """Initialize a ChatGLM2Config object.
        
        Args:
            self (ChatGLM2Config): An instance of the ChatGLM2Config class.
            num_layers (int, optional): The number of layers in the model. Defaults to 28.
            padded_vocab_size (int, optional): The size of the padded vocabulary. Defaults to 65024.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 4096.
            ffn_hidden_size (int, optional): The size of the feed-forward network hidden layers. Defaults to 13696.
            kv_channels (int, optional): The number of channels in the key-value attention. Defaults to 128.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 32.
            seq_length (int, optional): The maximum sequence length. Defaults to 2048.
            hidden_dropout (float, optional): The dropout probability for the hidden layers. Defaults to 0.0.
            classifier_dropout (float, optional): The dropout probability for the classifier layer. Defaults to None.
            attention_dropout (float, optional): The dropout probability for the attention layers. Defaults to 0.0.
            layernorm_epsilon (float, optional): The epsilon value for layer normalization. Defaults to 1e-05.
            rmsnorm (bool, optional): Whether to use RMSNorm for normalization. Defaults to True.
            apply_residual_connection_post_layernorm (bool, optional): Whether to apply residual connection after layer normalization. Defaults to False.
            post_layer_norm (bool, optional): Whether to apply layer normalization after each sublayer. Defaults to True.
            add_bias_linear (bool, optional): Whether to add bias to the linear layer. Defaults to False.
            add_qkv_bias (bool, optional): Whether to add bias to the query, key, and value layers. Defaults to False.
            bias_dropout_fusion (bool, optional): Whether to fuse bias dropout with linear layer. Defaults to True.
            multi_query_attention (bool, optional): Whether to use multi-query attention. Defaults to False.
            multi_query_group_num (int, optional): The number of groups for multi-query attention. Defaults to 1.
            apply_query_key_layer_scaling (bool, optional): Whether to apply scaling on query-key layer. Defaults to True.
            attention_softmax_in_fp32 (bool, optional): Whether to use FP32 for attention softmax. Defaults to True.
            fp32_residual_connection (bool, optional): Whether to use FP32 for residual connection. Defaults to False.
            quantization_bit (int, optional): The number of bits for quantization. Defaults to 0.
            pre_seq_len (int, optional): The length of the prefix sequence. Defaults to None.
            prefix_projection (bool, optional): Whether to use prefix projection. Defaults to False.
        
        Returns:
            None.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)

__all__ = ['ChatGLM2Config']
