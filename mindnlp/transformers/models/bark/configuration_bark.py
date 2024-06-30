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
"""
Bark config
"""
import os
from typing import Dict, Optional, Union

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


BARK_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "suno/bark-small": "https://hf-mirror.com/suno/bark-small/resolve/main/config.json",
    "suno/bark": "https://hf-mirror.com/suno/bark/resolve/main/config.json",
}

BARK_SUBMODELCONFIG_START_DOCSTRING = """
    This is the configuration class to store the configuration of a [`{model}`]. It is used to instantiate the model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Bark [suno/bark](https://hf-mirror.com/suno/bark)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        block_size (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        input_vocab_size (`int`, *optional*, defaults to 10_048):
            Vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`{model}`]. Defaults to 10_048 but should be carefully thought with
            regards to the chosen sub-model.
        output_vocab_size (`int`, *optional*, defaults to 10_048):
            Output vocabulary size of a Bark sub-model. Defines the number of different tokens that can be represented
            by the: `output_ids` when passing forward a [`{model}`]. Defaults to 10_048 but should be carefully thought
            with regards to the chosen sub-model.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the given sub-model.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer architecture.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the architecture.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        bias (`bool`, *optional*, defaults to `True`):
            Whether or not to use bias in the linear layers and layer norm layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
"""


class BarkSubModelConfig(PretrainedConfig):

    """
    BarkSubModelConfig represents the configuration class for the Bark sub-model.
    It inherits from PretrainedConfig and provides configuration parameters for the specific Bark sub-model.
    
    Attributes:
        block_size (int): The block size for the model.
        input_vocab_size (int): The size of the input vocabulary.
        output_vocab_size (int): The size of the output vocabulary.
        num_layers (int): The number of layers in the model.
        num_heads (int): The number of attention heads.
        hidden_size (int): The size of the hidden layers.
        dropout (float): The dropout probability.
        bias (bool): Indicates whether bias is used in the model.
        use_cache (bool): Indicates whether caching is enabled.
        initializer_range (float): The range for parameter initialization.

    Methods:
        from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, **kwargs):
            Method to create a BarkSubModelConfig instance from a pretrained model or path.

    Note:
        The from_pretrained method allows creating a BarkSubModelConfig instance from a pretrained model or path,
        with options to specify the cache directory, force download, and use of local files.
        The method also handles the configuration dictionary and checks for model type compatibility.
    """
    model_type = "bark_module"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "vocab_size": "input_vocab_size",
        "window_size": "block_size",
    }

    def __init__(
        self,
        block_size=1024,
        input_vocab_size=10_048,
        output_vocab_size=10_048,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        dropout=0.0,
        bias=True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        """
        Initializes a BarkSubModelConfig object.

        Args:
            block_size (int): The size of the block. Default is 1024.
            input_vocab_size (int): The size of the input vocabulary. Default is 10048.
            output_vocab_size (int): The size of the output vocabulary. Default is 10048.
            num_layers (int): The number of layers. Default is 12.
            num_heads (int): The number of heads. Default is 12.
            hidden_size (int): The size of the hidden layer. Default is 768.
            dropout (float): The dropout rate. Default is 0.0.
            bias (bool): Indicates whether bias is used. Default is True.
            initializer_range (float): The range for weight initialization. Default is 0.02.
            use_cache (bool): Indicates whether cache is used. Default is True.

        Returns:
            None.

        Raises:
            None
        """
        self.block_size = block_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        **kwargs,
    ) -> "PretrainedConfig":
        """
        This method is used to instantiate a 'BarkSubModelConfig' object from a pretrained model.

        Args:
            cls (class): The class object representing the 'BarkSubModelConfig' class.
            pretrained_model_name_or_path (Union[str, os.PathLike]): The name or path of the pretrained model.
            cache_dir (Optional[Union[str, os.PathLike]]): The directory to cache the downloaded files. Default is None.
            force_download (bool):
                Whether to force download the model files even if they already exist in the cache directory. Default is False.
            local_files_only (bool):
                Whether to use only the local files and not download any files if they are not available locally. Default is False.
            **kwargs: Additional keyword arguments.

        Returns:
            PretrainedConfig:
                An object of type 'PretrainedConfig' representing the instantiated 'BarkSubModelConfig'.

        Raises:
            None.
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the config dict if we are loading from Bark
        if config_dict.get("model_type") == "bark":
            config_dict = config_dict[f"{cls.model_type}_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BarkSemanticConfig(BarkSubModelConfig):

    """
    Represents a configuration class for semantic segmentation models in the Bark framework.
    This class inherits properties and methods from the BarkSubModelConfig class.

    Attributes:
        model_name (str): The name of the semantic segmentation model.
        num_classes (int): The number of classes in the semantic segmentation task.
        input_shape (tuple): The input shape of the model in the format (height, width, channels).
        backbone (str): The backbone architecture used in the model.
        pretrained_backbone (bool): Indicates if a pretrained backbone is used.
        normalization (str): The type of normalization applied to the input data.
        loss_function (str): The loss function used for training the model.
        optimizer (str): The optimizer used during model training.

    Methods:
        set_model_name:
            Sets the name of the semantic segmentation model.

        set_num_classes:
            Sets the number of classes in the semantic segmentation task.

        set_input_shape:
            Sets the input shape of the model.

        set_backbone:
            Sets the backbone architecture used in the model.

        set_pretrained_backbone:
            Sets whether a pretrained backbone is used.

        set_normalization:
            Sets the type of normalization applied to the input data.

        set_loss_function:
            Sets the loss function used for training the model.

        set_optimizer:
            Sets the optimizer used during model training.
    """
    model_type = "semantic"


class BarkCoarseConfig(BarkSubModelConfig):

    """
    BarkCoarseConfig is a Python class that represents the configuration settings for the coarse behavior model
    in the Bark autonomous driving simulation framework.
    This class inherits from the BarkSubModelConfig class.

    The BarkCoarseConfig class provides a set of parameters and options that can be used to configure the behavior of the coarse model.
    These parameters include settings related to the behavior model itself, such as the desired velocity,
    acceleration limits, and time horizons, as well as settings for the perception model,
    such as sensor range and field of view.

    Attributes:
        desired_velocity (float): The desired velocity of the ego vehicle.
        max_acceleration (float): The maximum acceleration limit for the ego vehicle.
        min_acceleration (float): The minimum acceleration limit for the ego vehicle.
        horizon_time (float): The time horizon for the behavior planning.
        perception_range (float): The range of the perception sensor.
        field_of_view (float): The field of view of the perception sensor.

    Methods:
        __init__(self, desired_velocity, max_acceleration, min_acceleration, horizon_time, perception_range, field_of_view):
            Initializes a new instance of the BarkCoarseConfig class with the specified parameters.
        get_desired_velocity(self): Returns the desired velocity of the ego vehicle.
        set_desired_velocity(self, desired_velocity): Sets the desired velocity of the ego vehicle.
        get_max_acceleration(self): Returns the maximum acceleration limit for the ego vehicle.
        set_max_acceleration(self, max_acceleration): Sets the maximum acceleration limit for the ego vehicle.
        get_min_acceleration(self): Returns the minimum acceleration limit for the ego vehicle.
        set_min_acceleration(self, min_acceleration): Sets the minimum acceleration limit for the ego vehicle.
        get_horizon_time(self): Returns the time horizon for the behavior planning.
        set_horizon_time(self, horizon_time): Sets the time horizon for the behavior planning.
        get_perception_range(self): Returns the range of the perception sensor.
        set_perception_range(self, perception_range): Sets the range of the perception sensor.
        get_field_of_view(self): Returns the field of view of the perception sensor.
        set_field_of_view(self, field_of_view): Sets the field of view of the perception sensor.

    """
    model_type = "coarse_acoustics"


class BarkFineConfig(BarkSubModelConfig):

    """
    BarkFineConfig represents the configuration settings for a fine-tuning model within the Bark framework.
    This class inherits from BarkSubModelConfig and provides parameters for configuring the fine-tuning process,
    including options for tying word embeddings, specifying the total number of codes, and the number of codes given.

    Parameters:
        tie_word_embeddings (bool): Flag indicating whether to tie word embeddings during fine-tuning.
        n_codes_total (int): The total number of codes used in the fine-tuning model.
        n_codes_given (int): The number of codes given as input to the fine-tuning model.

    Inherits from BarkSubModelConfig and initializes the configuration settings for the fine-tuning model based on the provided parameters.
    """
    model_type = "fine_acoustics"

    def __init__(self, tie_word_embeddings=True, n_codes_total=8, n_codes_given=1, **kwargs):
        """Initializes a new instance of the BarkFineConfig class.

        Args:
            self (BarkFineConfig): The object instance.
            tie_word_embeddings (bool): Whether to tie the word embeddings of the model. Defaults to True.
            n_codes_total (int): The total number of codes. Defaults to 8.
            n_codes_given (int): The number of given codes. Defaults to 1.

        Returns:
            None.

        Raises:
            None.
        """
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class BarkConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BarkModel`]. It is used to instantiate a Bark
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark
    [suno/bark](https://hf-mirror.com/suno/bark) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        semantic_config ([`BarkSemanticConfig`], *optional*):
            Configuration of the underlying semantic sub-model.
        coarse_acoustics_config ([`BarkCoarseConfig`], *optional*):
            Configuration of the underlying coarse acoustics sub-model.
        fine_acoustics_config ([`BarkFineConfig`], *optional*):
            Configuration of the underlying fine acoustics sub-model.
        codec_config ([`AutoConfig`], *optional*):
            Configuration of the underlying codec sub-model.

    Example:
        ```python
        >>> from transformers import (
        ...     BarkSemanticConfig,
        ...     BarkCoarseConfig,
        ...     BarkFineConfig,
        ...     BarkModel,
        ...     BarkConfig,
        ...     AutoConfig,
        ... )
        ...
        >>> # Initializing Bark sub-modules configurations.
        >>> semantic_config = BarkSemanticConfig()
        >>> coarse_acoustics_config = BarkCoarseConfig()
        >>> fine_acoustics_config = BarkFineConfig()
        >>> codec_config = AutoConfig.from_pretrained("facebook/encodec_24khz")
        ...
        ...
        >>> # Initializing a Bark module style configuration
        >>> configuration = BarkConfig.from_sub_model_configs(
        ...     semantic_config, coarse_acoustics_config, fine_acoustics_config, codec_config
        ... )
        ...
        >>> # Initializing a model (with random weights)
        >>> model = BarkModel(configuration)
        ...
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """
    model_type = "bark"

    def __init__(
        self,
        semantic_config: Dict = None,
        coarse_acoustics_config: Dict = None,
        fine_acoustics_config: Dict = None,
        codec_config: Dict = None,
        initializer_range=0.02,
        **kwargs,
    ):
        """
        Initializes a BarkConfig object with the provided configurations.
        
        Args:
            self: The instance of the BarkConfig class.
            semantic_config (Dict): Dictionary containing configuration for the semantic model. Defaults to None.
            coarse_acoustics_config (Dict): Dictionary containing configuration for the coarse acoustics model. Defaults to None.
            fine_acoustics_config (Dict): Dictionary containing configuration for the fine acoustics model. Defaults to None.
            codec_config (Dict): Dictionary containing configuration for the codec model. Defaults to None.
            initializer_range (float): Range for weight initialization. Defaults to 0.02.
        
        Returns:
            None.
        
        Raises:
            None
        """
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the semantic model with default values.")

        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info("coarse_acoustics_config is None. initializing the coarse model with default values.")

        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the fine model with default values.")

        if codec_config is None:
            codec_config = {}
            logger.info("codec_config is None. initializing the codec model with default values.")

        self.semantic_config = BarkSemanticConfig(**semantic_config)
        self.coarse_acoustics_config = BarkCoarseConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkFineConfig(**fine_acoustics_config)
        codec_model_type = codec_config["model_type"] if "model_type" in codec_config else "encodec"
        self.codec_config = CONFIG_MAPPING[codec_model_type](**codec_config)

        self.initializer_range = initializer_range

        super().__init__(**kwargs)

    @classmethod
    def from_sub_model_configs(
        cls,
        semantic_config: BarkSemanticConfig,
        coarse_acoustics_config: BarkCoarseConfig,
        fine_acoustics_config: BarkFineConfig,
        codec_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`BarkConfig`] (or a derived class) from bark sub-models configuration.

        Returns:
            [`BarkConfig`]: An instance of a configuration object
        """
        return cls(
            semantic_config=semantic_config.to_dict(),
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),
            fine_acoustics_config=fine_acoustics_config.to_dict(),
            codec_config=codec_config.to_dict(),
            **kwargs,
        )


__all__ = [
    "BARK_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "BarkCoarseConfig",
    "BarkConfig",
    "BarkFineConfig",
    "BarkSemanticConfig",
]
