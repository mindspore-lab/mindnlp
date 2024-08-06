# coding=utf-8
# Copyright 2022 Meta and The HuggingFace Inc. team. All rights reserved.
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
# ============================================================================
""" ESM model configuration"""

from dataclasses import asdict, dataclass
from typing import Optional

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)

# TODO Update this
ESM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/esm-1b": "https://hf-mirror.com/facebook/esm-1b/resolve/main/config.json",
    # See all ESM models at https://hf-mirror.com/models?filter=esm
}


class EsmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ESMModel`]. It is used to instantiate a ESM model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ESM
    [facebook/esm-1b](https://hf-mirror.com/facebook/esm-1b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the ESM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ESMModel`].
        mask_token_id (`int`, *optional*):
            The index of the mask token in the vocabulary. This must be included in the config because of the
            "mask-dropout" scaling trick, which will scale the inputs depending on the number of masked tokens.
        pad_token_id (`int`, *optional*):
            The index of the padding token in the vocabulary. This must be included in the config because certain parts
            of the ESM code use this instead of the attention mask.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 1026):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query", "rotary"`.
            For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        emb_layer_norm_before (`bool`, *optional*):
            Whether to apply layer normalization after embeddings but before the main stem of the network.
        token_dropout (`bool`, defaults to `False`):
            When this is enabled, masked tokens are treated as if they had been dropped out by input dropout.

    Example:
        ```python
        >>> from transformers import EsmModel, EsmConfig
        ...
        >>> # Initializing a ESM facebook/esm-1b style configuration >>> configuration = EsmConfig()
        ...
        >>> # Initializing a model from the configuration >>> model = ESMModel(configuration)
        ...
        >>> # Accessing the model configuration >>> configuration = model.config
        ```
    """
    model_type = "esm"

    def __init__(
        self,
        vocab_size=None,
        mask_token_id=None,
        pad_token_id=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1026,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_cache=True,
        emb_layer_norm_before=None,
        token_dropout=False,
        is_folding_model=False,
        esmfold_config=None,
        vocab_list=None,
        **kwargs,
    ):
        """
        Initializes an instance of the `EsmConfig` class.

        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to None.
            mask_token_id (int, optional): The ID of the mask token. Defaults to None.
            pad_token_id (int, optional): The ID of the padding token. Defaults to None.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 768.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 12.
            intermediate_size (int, optional): The size of the intermediate layers. Defaults to 3072.
            hidden_dropout_prob (float, optional): The dropout probability for hidden layers. Defaults to 0.1.
            attention_probs_dropout_prob (float, optional): The dropout probability for attention layers. Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum position embeddings. Defaults to 1026.
            initializer_range (float, optional): The range for initializer values. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-12.
            position_embedding_type (str, optional): The type of position embedding. Defaults to 'absolute'.
            use_cache (bool, optional): Whether to use cache. Defaults to True.
            emb_layer_norm_before (bool, optional): Whether to normalize embeddings before layers. Defaults to None.
            token_dropout (bool, optional): Whether to apply token dropout. Defaults to False.
            is_folding_model (bool, optional): Whether the model is a folding model. Defaults to False.
            esmfold_config (EsmFoldConfig, optional): The configuration for the folding model. Defaults to None.
            vocab_list (list, optional): The list of vocabulary tokens. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the HuggingFace port of ESMFold does not support `use_esm_attn_map`.
        """
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.emb_layer_norm_before = emb_layer_norm_before
        self.token_dropout = token_dropout
        self.is_folding_model = is_folding_model
        if is_folding_model:
            if esmfold_config is None:
                logger.info("No esmfold_config supplied for folding model, using default values.")
                esmfold_config = EsmFoldConfig()
            elif isinstance(esmfold_config, dict):
                esmfold_config = EsmFoldConfig(**esmfold_config)
            self.esmfold_config = esmfold_config
            if vocab_list is None:
                logger.warning("No vocab_list supplied for folding model, assuming the ESM-2 vocabulary!")
                self.vocab_list = get_default_vocab_list()
            else:
                self.vocab_list = vocab_list
        else:
            self.esmfold_config = None
            self.vocab_list = None
        if self.esmfold_config is not None and getattr(self.esmfold_config, "use_esm_attn_map", False):
            raise ValueError("The HuggingFace port of ESMFold does not support use_esm_attn_map at this time!")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()
        if isinstance(self.esmfold_config, EsmFoldConfig):
            output["esmfold_config"] = self.esmfold_config.to_dict()
        return output


@dataclass
class EsmFoldConfig:

    """
    Represents the configuration of an ESM (Efficient Speech Model) fold instance.

    This class provides methods to initialize the EsmFoldConfig instance and serialize it to a Python dictionary.

    The EsmFoldConfig class inherits from a base class and includes methods for post-initialization and dictionary serialization.

    Methods:
        __post_init__(self): Initializes the EsmFoldConfig instance, setting defaults for any missing attributes.
        to_dict(self): Serializes the EsmFoldConfig instance to a Python dictionary, including the trunk configuration.

    Attributes:
        trunk: Represents the configuration of the trunk model used in the ESM fold.

    Note:
        Ensure that the trunk attribute is either set to a TrunkConfig instance or a dictionary that can be converted to a TrunkConfig.

    Return:
        A Python dictionary containing all the attributes of the EsmFoldConfig instance, including the trunk configuration.
    """
    esm_type: str = None
    fp16_esm: bool = True
    use_esm_attn_map: bool = False
    esm_ablate_pairwise: bool = False
    esm_ablate_sequence: bool = False
    esm_input_dropout: float = 0

    embed_aa: bool = True
    bypass_lm: bool = False

    lddt_head_hid_dim: int = 128
    trunk: "TrunkConfig" = None

    def __post_init__(self):
        """
        The '__post_init__' method is used in the 'EsmFoldConfig' class to initialize the 'trunk' attribute.

        Args:
            self: An instance of the 'EsmFoldConfig' class.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method checks if the 'trunk' attribute is None. If it is, a new instance of the 'TrunkConfig' class
            is created and assigned to 'self.trunk'. If the 'trunk' attribute is of type dict, it is unpacked and
            passed as keyword arguments to create a new instance of the 'TrunkConfig' class,  which is then assigned to
            'self.trunk'. This method is typically called after the object is initialized to ensure that the 'trunk'
            attribute is properly set.

        Example:
            ```python
            >>> config = EsmFoldConfig()
            >>> config.__post_init__()
            >>> # The 'trunk' attribute will be initialized with a new instance of the 'TrunkConfig' class.
            ...
            >>> config = EsmFoldConfig(trunk={'option1': True, 'option2': False})
            >>> config.__post_init__()
            >>> # The 'trunk' attribute will be initialized with a new instance of the 'TrunkConfig' class,
            >>> # with 'option1' set to True and 'option2' set to False.
            ```
        """
        if self.trunk is None:
            self.trunk = TrunkConfig()
        elif isinstance(self.trunk, dict):
            self.trunk = TrunkConfig(**self.trunk)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = asdict(self)
        output["trunk"] = self.trunk.to_dict()
        return output


@dataclass
class TrunkConfig:

    """
    Represents the configuration settings for the Trunk model.
    This class defines the configuration attributes and their validations for the Trunk model.

    Attributes:
        structure_module (StructureModuleConfig): The configuration for the structure module.
        max_recycles (int): The maximum number of recycles, should be a positive integer.
        sequence_state_dim (int): The dimension of the sequence state.
        pairwise_state_dim (int): The dimension of the pairwise state.
        sequence_head_width (int): The width of the sequence head.
        pairwise_head_width (int): The width of the pairwise head.
        dropout (float): The dropout rate, should not be greater than 0.4.

    Raises:
        ValueError:
            If any of the following conditions are not met:

            - `max_recycles` is not a positive integer.
            - `sequence_state_dim` is not a round multiple of itself.
            - `pairwise_state_dim` is not a round multiple of itself.
            - `sequence_state_dim` is not equal to `sequence_num_heads * sequence_head_width`.
            - `pairwise_state_dim` is not equal to `pairwise_num_heads * pairwise_head_width`.
            - `pairwise_state_dim` is not an even number.
            - `dropout` is greater than 0.4.

    Methods:
        __post_init__(self): Performs post-initialization validations for the configuration attributes.
        to_dict(self): Serializes the instance to a Python dictionary, including the structure module configuration.

    Overrides:
        `~PretrainedConfig.to_dict`: Overrides the default `to_dict` method to include the structure module
        configuration in the dictionary output.
    """
    num_blocks: int = 48
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False
    max_recycles: int = 4
    chunk_size: Optional[int] = 128
    structure_module: "StructureModuleConfig" = None

    def __post_init__(self):
        """
        This method initializes the TrunkConfig class after its instantiation.

        Args:
            self: The instance of the TrunkConfig class.

        Returns:
            None.

        Raises:
            ValueError: If `max_recycles` is not a positive value.
            ValueError: If `sequence_state_dim` is not a round multiple of itself.
            ValueError: If `pairwise_state_dim` is not a round multiple of itself.
            ValueError: If `sequence_state_dim` is not equal to `sequence_num_heads * sequence_head_width`.
            ValueError: If `pairwise_state_dim` is not equal to `pairwise_num_heads * pairwise_head_width`.
            ValueError: If `pairwise_state_dim` is not an even number.
            ValueError: If `dropout` is greater than or equal to 0.4.
        """
        if self.structure_module is None:
            self.structure_module = StructureModuleConfig()
        elif isinstance(self.structure_module, dict):
            self.structure_module = StructureModuleConfig(**self.structure_module)

        if self.max_recycles <= 0:
            raise ValueError(f"`max_recycles` should be positive, got {self.max_recycles}.")
        if self.sequence_state_dim % self.sequence_state_dim != 0:
            raise ValueError(
                "`sequence_state_dim` should be a round multiple of `sequence_state_dim`, got"
                f" {self.sequence_state_dim} and {self.sequence_state_dim}."
            )
        if self.pairwise_state_dim % self.pairwise_state_dim != 0:
            raise ValueError(
                "`pairwise_state_dim` should be a round multiple of `pairwise_state_dim`, got"
                f" {self.pairwise_state_dim} and {self.pairwise_state_dim}."
            )

        sequence_num_heads = self.sequence_state_dim // self.sequence_head_width
        pairwise_num_heads = self.pairwise_state_dim // self.pairwise_head_width

        if self.sequence_state_dim != sequence_num_heads * self.sequence_head_width:
            raise ValueError(
                "`sequence_state_dim` should be equal to `sequence_num_heads * sequence_head_width, got"
                f" {self.sequence_state_dim} != {sequence_num_heads} * {self.sequence_head_width}."
            )
        if self.pairwise_state_dim != pairwise_num_heads * self.pairwise_head_width:
            raise ValueError(
                "`pairwise_state_dim` should be equal to `pairwise_num_heads * pairwise_head_width, got"
                f" {self.pairwise_state_dim} != {pairwise_num_heads} * {self.pairwise_head_width}."
            )
        if self.pairwise_state_dim % 2 != 0:
            raise ValueError(f"`pairwise_state_dim` should be even, got {self.pairwise_state_dim}.")

        if self.dropout >= 0.4:
            raise ValueError(f"`dropout` should not be greater than 0.4, got {self.dropout}.")

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = asdict(self)
        output["structure_module"] = self.structure_module.to_dict()
        return output


@dataclass
class StructureModuleConfig:
    """
    Args:
        sequence_dim:
            Single representation channel dimension
        pairwise_dim:
            Pair representation channel dimension
        ipa_dim:
            IPA hidden channel dimension
        resnet_dim:
            Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
        num_heads_ipa:
            Number of IPA heads
        num_qk_points:
            Number of query/key points to generate during IPA
        num_v_points:
            Number of value points to generate during IPA
        dropout_rate:
            Dropout rate used throughout the layer
        num_blocks:
            Number of structure module blocks
        num_transition_layers:
            Number of layers in the single representation transition (Alg. 23 lines 8-9)
        num_resnet_blocks:
            Number of blocks in the angle resnet
        num_angles:
            Number of angles to generate in the angle resnet
        trans_scale_factor:
            Scale of single representation transition hidden dimension
        epsilon:
            Small number used in angle resnet normalization
        inf:
            Large number used for attention masking
    """
    sequence_dim: int = 384
    pairwise_dim: int = 128
    ipa_dim: int = 16
    resnet_dim: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_transition_layers: int = 1
    num_resnet_blocks: int = 2
    num_angles: int = 7
    trans_scale_factor: int = 10
    epsilon: float = 1e-8
    inf: float = 1e5

    def to_dict(self):
        """
        Converts the current instance of the StructureModuleConfig class to a dictionary.

        Args:
            self (StructureModuleConfig): The current instance of the StructureModuleConfig class.

        Returns:
            dict: A dictionary representation of the current StructureModuleConfig instance.

        Raises:
            None.
        """
        return asdict(self)


def get_default_vocab_list():
    '''
    This function returns a list of default vocabulary items including special tokens and characters used in natural
    language processing tasks.

    Args:
        None.

    Returns:
        List:
            A list of default vocabulary items including '<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S',
            'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U',
            'Z', 'O', '.', '-', '<null_1>', '<mask>'.
    
    Raises:
        None.
    '''
    return (
        "<cls>",
        "<pad>",
        "<eos>",
        "<unk>",
        "L",
        "A",
        "G",
        "V",
        "S",
        "E",
        "R",
        "T",
        "I",
        "D",
        "P",
        "K",
        "Q",
        "N",
        "F",
        "Y",
        "M",
        "H",
        "W",
        "C",
        "X",
        "B",
        "U",
        "Z",
        "O",
        ".",
        "-",
        "<null_1>",
        "<mask>",
    )

__all__ = ["ESM_PRETRAINED_CONFIG_ARCHIVE_MAP", "EsmConfig"]
