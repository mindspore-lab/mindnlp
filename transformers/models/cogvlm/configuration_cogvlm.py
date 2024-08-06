"""
configuration
"""

try:
    from typing import Literal
except:
    from typing_extensions import Literal
from ...configuration_utils import PretrainedConfig


class CogVLMConfig(PretrainedConfig):

    """
    The `CogVLMConfig` class represents the configuration for a CogVLM (Cognitive Vision Language Model) model.
    It inherits from the `PretrainedConfig` class and provides a set of parameters to customize the
    behavior of the CogVLM model.

    Parameters:
        `vocab_size` (int, optional): The size of the vocabulary. Defaults to 32000.
        `hidden_size` (int, optional): The size of the hidden layers. Defaults to 4096.
        `intermediate_size` (int, optional): The size of the intermediate layers. Defaults to 11008.
        `num_hidden_layers` (int, optional): The number of hidden layers. Defaults to 32.
        `num_attention_heads` (int, optional): The number of attention heads. Defaults to 32.
        `hidden_act` (str, optional): The activation function for the hidden layers. Defaults to 'silu'.
        `max_position_embeddings` (int, optional): The maximum number of position embeddings. Defaults to 2048.
        `initializer_range` (float, optional): The range for the weight initialization. Defaults to 0.02.
        `rms_norm_eps` (float, optional): The epsilon value for the RMS normalization. Defaults to 1e-06.
        `template_version` (Literal['base', 'chat'], optional): The template version to use. Defaults to 'chat'.
        `pad_token_id` (int, optional): The token ID for padding. Defaults to 0.
        `bos_token_id` (int, optional): The token ID for the beginning of sentence. Defaults to 1.
        `eos_token_id` (int, optional): The token ID for the end of sentence. Defaults to 2.
        `tie_word_embeddings` (bool, optional): Whether to tie the word embeddings. Defaults to False.
        `use_cache` (bool, optional): Whether to use cache during model inference. Defaults to True.

    Attributes:
        `hidden_size` (int): The size of the hidden layers.
        `intermediate_size` (int): The size of the intermediate layers.
        `num_attention_heads` (int): The number of attention heads.
        `max_position_embeddings` (int): The maximum number of position embeddings.
        `rms_norm_eps` (float): The epsilon value for the RMS normalization.
        `initializer_range` (float): The range for the weight initialization.
        `vocab_size` (int): The size of the vocabulary.
        `num_hidden_layers` (int): The number of hidden layers.
        `hidden_act` (str): The activation function for the hidden layers.
        `template_version` (Literal['base', 'chat']): The template version to use.
        `use_cache` (bool): Whether to use cache during model inference.
        `vision_config` (dict): The configuration for the vision module.
            The `vision_config` dictionary contains the following keys.

            - `dropout_prob` (float): The dropout probability for the vision module.
            - `hidden_act` (str): The activation function for the vision module.
            - `hidden_size` (int): The size of the hidden layers in the vision module.
            - `image_size` (int): The size of the input images.
            - `in_channels` (int): The number of input channels.
            - `intermediate_size` (int): The size of the intermediate layers in the vision module.
            - `layer_norm_eps` (float): The epsilon value for layer normalization in the vision module.
            - `num_heads` (int): The number of attention heads in the vision module.
            - `num_hidden_layers` (int): The number of hidden layers in the vision module.
            - `num_positions` (int): The maximum number of positions in the vision module.
            - `patch_size` (int): The size of the patches in the vision module.
    
    Note:
        This class does not include the actual model architecture, but only the configuration parameters for the CogVLM model.
    """
    _auto_class = "AutoConfig"

    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            hidden_act='silu',
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-06,
            template_version: Literal["base", "chat"] = "chat",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            use_cache=True,
            **kwargs,
    ):
        """
        Initialize CogVLMConfig.
        
        Args:
            self: The instance of the class.
            vocab_size (int, optional): The size of the vocabulary. Defaults to 32000.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 4096.
            intermediate_size (int, optional): The size of the intermediate layer in the transformer. Defaults to 11008.
            num_hidden_layers (int, optional): The number of hidden layers in the transformer. Defaults to 32.
            num_attention_heads (int, optional): The number of attention heads in the transformer. Defaults to 32.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'silu'.
            max_position_embeddings (int, optional): The maximum position for positional embeddings. Defaults to 2048.
            initializer_range (float, optional): The range for weight initialization. Defaults to 0.02.
            rms_norm_eps (float, optional): The epsilon value for RMS normalization. Defaults to 1e-06.
            template_version (Literal['base', 'chat'], optional): The version of the template. Defaults to 'chat'.
            pad_token_id (int, optional): The id for padding token. Defaults to 0.
            bos_token_id (int, optional): The id for beginning of sequence token. Defaults to 1.
            eos_token_id (int, optional): The id for end of sequence token. Defaults to 2.
            tie_word_embeddings (bool, optional): Whether to tie word embeddings. Defaults to False.
            use_cache (bool, optional): Whether to use caching. Defaults to True.
        
        Returns:
            None.
        
        Raises:
            TypeError: If vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads,
                max_position_embeddings, pad_token_id, bos_token_id, eos_token_id are not integers.
            ValueError: If initializer_range, rms_norm_eps are not floats, or if template_version is not 'base' or 'chat'.
            AssertionError: If hidden_act is not a string.
            NotImplementedError: If tie_word_embeddings is not a boolean or if use_cache is not a boolean.
        """
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.template_version = template_version
        self.use_cache = use_cache
        self.vision_config = {
            "dropout_prob": 0.0,
            "hidden_act": "gelu",
            "hidden_size": 1792,
            "image_size": 490,
            "in_channels": 3,
            "intermediate_size": 15360,
            "layer_norm_eps": 1e-06,
            "num_heads": 16,
            "num_hidden_layers": 63,
            "num_positions": 1226,
            "patch_size": 14
        }
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

__all__ = ['CogVLMConfig']
