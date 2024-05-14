"""
configuration
"""

from typing import Literal
from ...configuration_utils import PretrainedConfig


class CogVLMConfig(PretrainedConfig):
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
