# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Blip model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ....utils import logging


logger = logging.get_logger(__name__)


class BlipTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipTextModel`]. It is used to instantiate a BLIP
    text model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the `BlipText` used by the [base
    architectures](https://huggingface.co/Salesforce/blip-vqa-base).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 30524):
            Vocabulary size of the `Blip` text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`BlipModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        encoder_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers from the vision model.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        bos_token_id (`int`, *optional*, defaults to 30522):
            The id of the `beginning-of-sequence` token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the `end-of-sequence` token.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the `padding` token.
        sep_token_id (`int`, *optional*, defaults to 102):
            The id of the `separator` token.
        is_decoder (`bool`, *optional*, defaults to `True`):
            Whether the model is used as a decoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        label_smoothing (float, *optional*):
            A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.

    Example:
        ```python
        >>> from transformers import BlipTextConfig, BlipTextModel
        ...
        >>> # Initializing a BlipTextConfig with Salesforce/blip-vqa-base style configuration
        >>> configuration = BlipTextConfig()
        ...
        >>> # Initializing a BlipTextModel (with random weights) from the Salesforce/blip-vqa-base style configuration
        >>> model = BlipTextModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "blip_text_model"

    def __init__(
        self,
        vocab_size=30524,
        hidden_size=768,
        encoder_hidden_size=768,
        intermediate_size=3072,
        projection_dim=768,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=512,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        bos_token_id=30522,
        eos_token_id=2,
        pad_token_id=0,
        sep_token_id=102,
        is_decoder=True,
        use_cache=True,
        label_smoothing=0.0,
        **kwargs,
    ):
        """
        Initializes a BlipTextConfig object with the given parameters.

        Args:
            self: The BlipTextConfig instance.
            vocab_size (int, optional): The size of the vocabulary. Default is 30524.
            hidden_size (int, optional): The size of the hidden layers. Default is 768.
            encoder_hidden_size (int, optional): The size of the encoder hidden layers. Default is 768.
            intermediate_size (int, optional): The size of the intermediate layers. Default is 3072.
            projection_dim (int, optional): The projection dimension. Default is 768.
            num_hidden_layers (int, optional): The number of hidden layers. Default is 12.
            num_attention_heads (int, optional): The number of attention heads. Default is 8.
            max_position_embeddings (int, optional): The maximum position embeddings. Default is 512.
            hidden_act (str, optional): The activation function for the hidden layers. Default is 'gelu'.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Default is 1e-12.
            hidden_dropout_prob (float, optional): The dropout probability for the hidden layers. Default is 0.0.
            attention_probs_dropout_prob (float, optional): The dropout probability for attention probabilities. Default is 0.0.
            initializer_range (float, optional): The range for weight initialization. Default is 0.02.
            bos_token_id (int, optional): The ID of the beginning of sentence token. Default is 30522.
            eos_token_id (int, optional): The ID of the end of sentence token. Default is 2.
            pad_token_id (int, optional): The ID of the padding token. Default is 0.
            sep_token_id (int, optional): The ID of the separator token. Default is 102.
            is_decoder (bool, optional): Whether the model is a decoder. Default is True.
            use_cache (bool, optional): Whether to use cache for faster decoding. Default is True.
            label_smoothing (float, optional): The label smoothing factor. Default is 0.0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            sep_token_id=sep_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.is_decoder = is_decoder
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        """
        This method 'from_pretrained' in the class 'BlipTextConfig' is used to instantiate a model configuration object
        based on a pretrained model name or path.

        Args:
            cls (class): The class object itself.
            pretrained_model_name_or_path (Union[str, os.PathLike]): A string representing the name of a pretrained model
                or a valid path to a pretrained model configuration file. This parameter is mandatory and required for
                initializing the configuration object.

        Returns:
            PretrainedConfig: An instance of 'PretrainedConfig' class representing the configuration settings of
                the pretrained model.
                The method returns the configuration object based on the provided pretrained model name or path.

        Raises:
            TypeError: If the provided 'pretrained_model_name_or_path' is not a string or a valid path-like object.
            KeyError: If the 'model_type' key is missing from the configuration dictionary.
            Warning: If the model type in the configuration dictionary does not match the class model type,
                a warning message is logged as this may lead to errors during instantiation.
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from BlipConfig
        if config_dict.get("model_type") == "blip":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BlipVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BlipVisionModel`]. It is used to instantiate a
    BLIP vision model according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the Blip-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 384):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:
        ```python
        >>> from transformers import BlipVisionConfig, BlipVisionModel
        ...
        >>> # Initializing a BlipVisionConfig with Salesforce/blip-vqa-base style configuration
        >>> configuration = BlipVisionConfig()
        ...
        >>> # Initializing a BlipVisionModel (with random weights) from the Salesforce/blip-vqa-base style configuration
        >>> model = BlipVisionModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "blip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        image_size=384,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=1e-10,
        **kwargs,
    ):
        '''
        Initializes a BlipVisionConfig instance.

        Args:
            self: The object instance.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 768.
            intermediate_size (int, optional): The size of the intermediate layers. Defaults to 3072.
            projection_dim (int, optional): The dimension of the projected output. Defaults to 512.
            num_hidden_layers (int, optional): The number of hidden layers. Defaults to 12.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 12.
            image_size (int, optional): The size of the input image. Defaults to 384.
            patch_size (int, optional): The size of the image patch. Defaults to 16.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'gelu'.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-05.
            attention_dropout (float, optional): The dropout rate for attention layers. Defaults to 0.0.
            initializer_range (float, optional): The range for parameter initialization. Defaults to 1e-10.

        Returns:
            None.

        Raises:
            None.
        '''
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        """
        This method 'from_pretrained' in the class 'BlipVisionConfig' is used to create a new instance of the class by loading a pretrained model configuration.

        Args:
            cls (class): The class object itself, automatically passed as the first argument.
            pretrained_model_name_or_path (Union[str, os.PathLike]): A string representing the name or path of
                the pretrained model. It can be either a string or a PathLike object. This parameter is used to
                fetch the configuration dictionary for the pretrained model.

        Returns:
            PretrainedConfig: An instance of the 'PretrainedConfig' class representing the configuration of
                the pretrained model. The method returns this configuration for further use.

        Raises:
            None:
                No specific exceptions are documented to be raised by this method based on the provided code snippet.
                However, potential exceptions might include:

                - KeyError: If the 'model_type' key is missing in the configuration dictionary.
                - AttributeError: If the 'model_type' attribute is not present in the class.
                - Warning: If the model type being used is different from the expected model type, a warning message is logged.

        Note:
            It is recommended to handle exceptions that may occur during the execution of this method to
            ensure proper error handling and flow control.
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from BlipConfig
        if config_dict.get("model_type") == "blip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BlipConfig(PretrainedConfig):
    r"""
    [`BlipConfig`] is the configuration class to store the configuration of a [`BlipModel`]. It is used to instantiate
    a BLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the BLIP-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original BLIP implementation.
        image_text_hidden_size (`int`, *optional*, defaults to 256):
            Dimentionality of the hidden state of the image-text fusion layer.
        label_smoothing (float, optional, *optional*, defaults to 0.0):
            A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:
        ```python
        >>> from transformers import BlipConfig, BlipModel
        ...
        >>> # Initializing a BlipConfig with Salesforce/blip-vqa-base style configuration
        >>> configuration = BlipConfig()
        ...
        >>> # Initializing a BlipPModel (with random weights) from the Salesforce/blip-vqa-base style configuration
        >>> model = BlipModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ...
        >>> # We can also initialize a BlipConfig from a BlipTextConfig and a BlipVisionConfig
        ...
        >>> # Initializing a BLIPText and BLIPVision configuration
        >>> config_text = BlipTextConfig()
        >>> config_vision = BlipVisionConfig()
        ...
        >>> config = BlipConfig.from_text_vision_configs(config_text, config_vision)
        ```
    """
    model_type = "blip"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        image_text_hidden_size=256,
        label_smoothing=0.0,
        **kwargs,
    ):
        """
        Initializes a BlipConfig object with the provided parameters.
        
        Args:
            self (BlipConfig): The instance of the BlipConfig class.
            text_config (dict, optional): Configuration parameters for text. Defaults to None.
            vision_config (dict, optional): Configuration parameters for vision. Defaults to None.
            projection_dim (int, optional): The dimension of the projection. Defaults to 512.
            logit_scale_init_value (float, optional): The initial value for logit scaling. Defaults to 2.6592.
            image_text_hidden_size (int, optional): The size of the hidden layer for image and text. Defaults to 256.
            label_smoothing (float, optional): The amount of label smoothing. Defaults to 0.0.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `BlipTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. Initializing the `BlipVisionConfig` with default values.")

        self.text_config = BlipTextConfig(**text_config)
        self.vision_config = BlipVisionConfig(**vision_config)

        self.text_config.encoder_hidden_size = self.vision_config.hidden_size

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing

    @classmethod
    def from_text_vision_configs(cls, text_config: BlipTextConfig, vision_config: BlipVisionConfig, **kwargs):
        r"""
        Instantiate a [`BlipConfig`] (or a derived class) from blip text model configuration and blip vision model
        configuration.

        Returns:
            [`BlipConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

__all__ = ['BlipConfig', 'BlipVisionConfig', 'BlipTextConfig']
