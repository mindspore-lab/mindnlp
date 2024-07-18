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
""" Moss model configuration"""

from ...configuration_utils import PretrainedConfig


__all__ = ['MossConfig']

Moss_SUPPORT_LIST = ["moss"]


class MossConfig(PretrainedConfig):
    """
    Configuration for moss
    """
    model_type = "moss"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    def __init__(
            self,
            vocab_size=107008,
            n_positions=2048,
            n_ctx=2048,
            n_embd=4096,
            n_layer=28,
            n_head=16,
            rotary_dim=64,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=106028,
            eos_token_id=106068,
            tie_word_embeddings=False,
            wbits=32,
            groupsize=128,
            # max_position_embeddings = 1024,
            **kwargs,
    ):
        """
        Initialize a MossConfig object.
        
        Args:
            vocab_size (int): The size of the vocabulary.
            n_positions (int): The number of positions.
            n_ctx (int): The context size.
            n_embd (int): The embedding size.
            n_layer (int): The number of layers.
            n_head (int): The number of attention heads.
            rotary_dim (int): The dimension for rotary embeddings.
            n_inner (int): The inner dimension size (if applicable).
            activation_function (str): The activation function used.
            resid_pdrop (float): The dropout probability for residual connections.
            embd_pdrop (float): The dropout probability for embeddings.
            attn_pdrop (float): The dropout probability for attention layers.
            layer_norm_epsilon (float): The epsilon value for layer normalization.
            initializer_range (float): The range for parameter initialization.
            use_cache (bool): Flag indicating whether to use cache.
            bos_token_id (int): The ID for the beginning of sequence token.
            eos_token_id (int): The ID for the end of sequence token.
            tie_word_embeddings (bool): Flag indicating whether word embeddings should be tied.
            wbits (int): The number of bits for weight quantization.
            groupsize (int): The group size for quantization.

        Returns:
            None

        Raises:
            ValueError: If an invalid parameter value is provided.
            TypeError: If the input types are incorrect.
            RuntimeError: If an unexpected error occurs during initialization.
        """
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.wbits = wbits
        self.groupsize = groupsize
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_position_embeddings = n_positions
        self.hidden_size = n_embd
        self.num_attention_heads=n_head
        self.num_hidden_layers=n_layer
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
