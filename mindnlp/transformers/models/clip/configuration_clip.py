# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" CLIP model configuration"""

import os
from typing import Union

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)

CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/clip-vit-base-patch32": "https://hf-mirror.com/openai/clip-vit-base-patch32/resolve/main/config.json",
    # See all CLIP models at https://hf-mirror.com/models?filter=clip
}


class CLIPTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPTextModel`]. It is used to instantiate a CLIP
    text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the text encoder of the CLIP
    [openai/clip-vit-base-patch32](https://hf-mirror.com/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 49408):
            Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`CLIPModel`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 77):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        pad_token_id (`int`, *optional*, defaults to 1):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 49406):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 49407):
            End of stream token id.

    Example:
        ```python
        >>> from transformers import CLIPTextConfig, CLIPTextModel
        ...
        >>> # Initializing a CLIPTextConfig with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPTextConfig()
        ...
        >>> # Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPTextModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "clip_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        # This differs from `CLIPTokenizer`'s default and from openai/clip
        # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        """
        Initialize CLIPTextConfig.

        Args:
            vocab_size (int, optional): The size of the vocabulary. Default is 49408.
            hidden_size (int, optional): The size of the hidden layers. Default is 512.
            intermediate_size (int, optional): The size of the intermediate layers. Default is 2048.
            projection_dim (int, optional): The projection dimension. Default is 512.
            num_hidden_layers (int, optional): The number of hidden layers. Default is 12.
            num_attention_heads (int, optional): The number of attention heads. Default is 8.
            max_position_embeddings (int, optional): The maximum position embeddings. Default is 77.
            hidden_act (str, optional): The type of activation function for the hidden layers. Default is 'quick_gelu'.
            layer_norm_eps (float, optional): Epsilon value for layer normalization. Default is 1e-05.
            attention_dropout (float, optional): The dropout rate for attention layers. Default is 0.0.
            initializer_range (float, optional): The range for parameter initializers. Default is 0.02.
            initializer_factor (float, optional): The factor for parameter initializers. Default is 1.0.
            pad_token_id (int, optional): The ID of the padding token. Default is 1.
            bos_token_id (int, optional): The ID of the beginning of sequence token. Default is 49406.
            eos_token_id (int, optional): The ID of the end of sequence token. Default is 49407.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        """
        Creates a CLIPTextConfig instance from a pretrained model.

        Args:
            cls (type): The class object.
            pretrained_model_name_or_path (Union[str, os.PathLike]): The name or path of the pretrained model.

        Returns:
            PretrainedConfig: A CLIPTextConfig instance initialized with the configuration specified by the pretrained model.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the configuration dictionary does not contain the required information.
            Warning: If the model type being used for instantiation does not match the class's model type, which may lead to errors.
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLIPVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPVisionModel`]. It is used to instantiate a
    CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP
    [openai/clip-vit-base-patch32](https://hf-mirror.com/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:
        ```python
        >>> from transformers import CLIPVisionConfig, CLIPVisionModel
        ...
        >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPVisionConfig()
        ...
        >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPVisionModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        """
        Initialize a CLIPVisionConfig object with the provided configuration parameters.

        Args:
            hidden_size (int): The size of the hidden layers in the network.
            intermediate_size (int): The size of the intermediate hidden layers in the network.
            projection_dim (int): The dimension of the projected embeddings.
            num_hidden_layers (int): The number of hidden layers in the network.
            num_attention_heads (int): The number of attention heads in the network.
            num_channels (int): The number of channels in the input image.
            image_size (int): The size of the input image.
            patch_size (int): The size of the image patch used in the network.
            hidden_act (str): The activation function used in the hidden layers.
            layer_norm_eps (float): The epsilon value for layer normalization.
            attention_dropout (float): The dropout rate for attention layers.
            initializer_range (float): The range for parameter initialization.
            initializer_factor (float): The factor for parameter initialization.

        Returns:
            None.

        Raises:
            ValueError: If any of the input parameters are invalid or out of range.
        """
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        """
        Load a pretrained configuration from a given model name or path.

        Args:
            cls (class): The class object.
            pretrained_model_name_or_path (Union[str, os.PathLike]): The name or path of the pretrained model.
                It can be either a string representing the name of the model or a path-like object pointing to the model location.

        Returns:
            PretrainedConfig: The loaded pretrained configuration.

        Raises:
            None.

        This method is a class method that allows loading a pretrained configuration. It takes in the class object 'cls'
        and the name or path of the pretrained model 'pretrained_model_name_or_path' as parameters. The method returns an instance
        of type 'PretrainedConfig', which represents the loaded pretrained configuration.

        The 'pretrained_model_name_or_path' parameter can be either a string representing the name of the pretrained model
        or a path-like object pointing to the location of the model. It is used to identify and locate the pretrained model
        that needs to be loaded.

        Note: If the loaded configuration belongs to the 'clip' model type, the 'config_dict' will be updated to use the
        'vision_config' sub-dictionary. Additionally, if the 'model_type' attribute is present in the 'cls' class and
        the loaded configuration's 'model_type' is different from 'cls.model_type', a warning will be logged indicating
        that instantiating a model of different types may lead to errors.

        Example:
            ```python
            >>> config = CLIPVisionConfig.from_pretrained("clip_model")
            ...
            ```
            In the above example, the 'from_pretrained' method is called on the 'CLIPVisionConfig' class to load the pretrained
            configuration of the 'clip_model'. The resulting configuration is stored in the 'config' variable.
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CLIPConfig(PretrainedConfig):
    r"""
    [`CLIPConfig`] is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate
    a CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://hf-mirror.com/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:
        ```python
        >>> from transformers import CLIPConfig, CLIPModel
        ...
        >>> # Initializing a CLIPConfig with openai/clip-vit-base-patch32 style configuration
        >>> configuration = CLIPConfig()
        ...
        >>> # Initializing a CLIPModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
        >>> model = CLIPModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ...
        >>> # We can also initialize a CLIPConfig from a CLIPTextConfig and a CLIPVisionConfig
        >>> from transformers import CLIPTextConfig, CLIPVisionConfig
        ...
        >>> # Initializing a CLIPText and CLIPVision configuration
        >>> config_text = CLIPTextConfig()
        >>> config_vision = CLIPVisionConfig()
        ...
        >>> config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
        ```
    """
    model_type = "clip"

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        """
        Initializes a new instance of CLIPConfig.

        Args:
            self: The instance of the class.
            text_config (dict): The configuration for text inputs. If provided, overrides default values. Default is None.
            vision_config (dict): The configuration for vision inputs. If provided, overrides default values. Default is None.
            projection_dim (int): The dimension of the projection. Default is 512.
            logit_scale_init_value (float): The initial value for logit scaling. Default is 2.6592.

        Returns:
            None

        Raises:
            TypeError: If text_config or vision_config are not of type dict.
            ValueError: If projection_dim or logit_scale_init_value are not of type int or float respectively.
            KeyError: If 'transformers_version' key is present in text_config or vision_config.
            AttributeError: If 'id2label' key is not present in vision_config.
        """
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid them being saved (which causes a lot
        # of confusion!).
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)

        super().__init__(**kwargs)

        # Instead of simply assigning `[text|vision]_config_dict` to `[text|vision]_config`, we use the values in
        # `[text|vision]_config_dict` to update the values in `[text|vision]_config`. The values should be same in most
        # cases, but we don't want to break anything regarding `_config_dict` that existed before commit `8827e1b2`.
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = CLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and `text_config` but being different.
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and key not in ["transformers_version"]:
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f"`{key}` is found in both `text_config_dict` and `text_config` but with different values. "
                            f'The value `text_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`. The "
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = CLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if "id2label" in _vision_config_dict:
                _vision_config_dict["id2label"] = {
                    str(key): value for key, value in _vision_config_dict["id2label"].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict` and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and key not in ["transformers_version"]:
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f"`{key}` is found in both `vision_config_dict` and `vision_config` but with different "
                            f'values. The value `vision_config_dict["{key}"]` will be used instead.'
                        )
                    # If inferred from default argument values (just to be super careful)
                    else:
                        message = (
                            f"`vision_config_dict` is provided which will be used to initialize `CLIPVisionConfig`. "
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `CLIPTextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `CLIPVisionConfig` with default values.")

        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: CLIPTextConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`CLIPConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`CLIPConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

__all__ = [
    "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "CLIPConfig",
    "CLIPTextConfig",
    "CLIPVisionConfig",
]
