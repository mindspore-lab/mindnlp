# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" CodeGen model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/codegen-350M-nl": "https://hf-mirror.com/Salesforce/codegen-350M-nl/resolve/main/config.json",
    "Salesforce/codegen-350M-multi": "https://hf-mirror.com/Salesforce/codegen-350M-multi/resolve/main/config.json",
    "Salesforce/codegen-350M-mono": "https://hf-mirror.com/Salesforce/codegen-350M-mono/resolve/main/config.json",
    "Salesforce/codegen-2B-nl": "https://hf-mirror.com/Salesforce/codegen-2B-nl/resolve/main/config.json",
    "Salesforce/codegen-2B-multi": "https://hf-mirror.com/Salesforce/codegen-2B-multi/resolve/main/config.json",
    "Salesforce/codegen-2B-mono": "https://hf-mirror.com/Salesforce/codegen-2B-mono/resolve/main/config.json",
    "Salesforce/codegen-6B-nl": "https://hf-mirror.com/Salesforce/codegen-6B-nl/resolve/main/config.json",
    "Salesforce/codegen-6B-multi": "https://hf-mirror.com/Salesforce/codegen-6B-multi/resolve/main/config.json",
    "Salesforce/codegen-6B-mono": "https://hf-mirror.com/Salesforce/codegen-6B-mono/resolve/main/config.json",
    "Salesforce/codegen-16B-nl": "https://hf-mirror.com/Salesforce/codegen-16B-nl/resolve/main/config.json",
    "Salesforce/codegen-16B-multi": "https://hf-mirror.com/Salesforce/codegen-16B-multi/resolve/main/config.json",
    "Salesforce/codegen-16B-mono": "https://hf-mirror.com/Salesforce/codegen-16B-mono/resolve/main/config.json",
}


class CodeGenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CodeGenModel`]. It is used to instantiate a
    CodeGen model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CodeGen
    [Salesforce/codegen-2B-mono](https://hf-mirror.com/Salesforce/codegen-2B-mono) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50400):
            Vocabulary size of the CodeGen model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`CodeGenModel`].
        n_positions (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_ctx (`int`, *optional*, defaults to 2048):
            This attribute is used in `CodeGenModel.__init__` without any real effect.
        n_embd (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        rotary_dim (`int`, *optional*, defaults to 64):
            Number of dimensions in the embedding that Rotary Position Embedding is applied to.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        activation_function (`str`, *optional*, defaults to `"gelu_new"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 50256):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.

    Example:
        ```python
        >>> from transformers import CodeGenConfig, CodeGenModel
        ...
        >>> # Initializing a CodeGen 6B configuration
        >>> configuration = CodeGenConfig()
        ...
        >>> # Initializing a model (with random weights) from the configuration
        >>> model = CodeGenModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "codegen"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50400,
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
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        **kwargs,
    ):
        """
        Initializes an instance of the CodeGenConfig class.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 50400.
            n_positions (int, optional): The number of positions. Defaults to 2048.
            n_ctx (int, optional): The context size. Defaults to 2048.
            n_embd (int, optional): The embedding size. Defaults to 4096.
            n_layer (int, optional): The number of layers. Defaults to 28.
            n_head (int, optional): The number of attention heads. Defaults to 16.
            rotary_dim (int, optional): The dimension for rotary positional embeddings. Defaults to 64.
            n_inner (int, optional): The inner size of the feed-forward networks. Defaults to None.
            activation_function (str, optional): The activation function to use. Defaults to 'gelu_new'.
            resid_pdrop (float, optional): The dropout rate for residual connections. Defaults to 0.0.
            embd_pdrop (float, optional): The dropout rate for embeddings. Defaults to 0.0.
            attn_pdrop (float, optional): The dropout rate for attention probabilities. Defaults to 0.0.
            layer_norm_epsilon (float, optional): The epsilon value for layer normalization. Defaults to 1e-05.
            initializer_range (float, optional): The range for weight initialization. Defaults to 0.02.
            use_cache (bool, optional): Whether to use caching. Defaults to True.
            bos_token_id (int, optional): The ID of the beginning-of-sentence token. Defaults to 50256.
            eos_token_id (int, optional): The ID of the end-of-sentence token. Defaults to 50256.
            tie_word_embeddings (bool, optional): Whether to tie word embeddings. Defaults to False.
        
        Returns:
            None.
        
        Raises:
            None.
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

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

__all__ = ['CodeGenConfig']
