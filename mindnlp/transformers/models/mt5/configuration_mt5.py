# coding=utf-8
# Copyright 2020, The T5 Authors and HuggingFace Inc.
#
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
""" mT5 model configuration"""
from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


class MT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MT5Model`] or a [`TFMT5Model`]. It is used to
    instantiate a mT5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the mT5
    [google/mt5-small](https://hf-mirror.com/google/mt5-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Arguments:
        vocab_size (`int`, *optional*, defaults to 250112):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. In the conventional context,
            it is typically expected that `d_kv` has to be equal to `d_model // num_heads`.
            But in the architecture of mt5-small, `d_kv` is not equal to `d_model //num_heads`.
            The `inner_dim` of the projection layer will be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 1024):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    """
    model_type = "mt5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=250112,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=8,
        num_decoder_layers=None,
        num_heads=6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        tokenizer_class="T5Tokenizer",
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        classifier_dropout=0.0,
        **kwargs,
    ):
        """
        Initialize a new MT5Config object.
        
        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimension of the model.
            d_kv (int): The dimension of the key and value vectors in the attention mechanism.
            d_ff (int): The dimension of the feed-forward network.
            num_layers (int): The total number of layers in the model.
            num_decoder_layers (int): The number of decoder layers. If None, it defaults to the total number of layers.
            num_heads (int): The number of attention heads in the multi-head attention mechanism.
            relative_attention_num_buckets (int): The number of buckets for relative position embeddings.
            relative_attention_max_distance (int): The maximum distance for relative position embeddings.
            dropout_rate (float): The dropout rate for regularization.
            layer_norm_epsilon (float): The epsilon value for layer normalization.
            initializer_factor (float): The factor for initializing parameters.
            feed_forward_proj (str): The type of feed-forward projection function.
            is_encoder_decoder (bool): Indicates whether the model is an encoder-decoder architecture.
            use_cache (bool): Indicates whether caching is enabled.
            tokenizer_class (str): The class name of the tokenizer.
            tie_word_embeddings (bool): Indicates whether word embeddings are tied in the model.
            pad_token_id (int): The token ID for padding.
            eos_token_id (int): The token ID for end-of-sequence.
            decoder_start_token_id (int): The token ID for the decoder start token.
            classifier_dropout (float): The dropout rate for the classifier.

        Returns:
            None

        Raises:
            ValueError: If the feed_forward_proj format is invalid or not supported.
        """
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            tokenizer_class=tokenizer_class,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

    @property
    def hidden_size(self):
        """
        Returns the hidden size of the MT5Config.

        Args:
            self: An instance of the MT5Config class.

        Returns:
            int:
                The hidden size (d_model) of the MT5Config.
        
        Raises:
            None.
        """
        return self.d_model

    @property
    def num_attention_heads(self):
        """
        This method, num_attention_heads, is a property method within the MT5Config class, and it takes one parameter,
        self.
        
        Args:
            self (object): A reference to the instance of the class.
            
        Returns:
            num_heads: This method returns the value of 'num_heads' attribute.
        
        Raises:
            None.
        """
        return self.num_heads

    @property
    def num_hidden_layers(self):
        '''
        Returns the number of hidden layers in the MT5Config.
        
        Args:
            self: The instance of the MT5Config class.
        
        Returns:
            None
        
        Raises:
            None.
        '''
        return self.num_layers

__all__ = ['MT5Config']
