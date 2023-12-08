# coding=utf-8
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

""" Falcon configuration"""
from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tiiuae/falcon-40b": "https://huggingface.co/tiiuae/falcon-40b/resolve/main/config.json",
    "tiiuae/falcon-7b": "https://huggingface.co/tiiuae/falcon-7b/resolve/main/config.json",
}

logger = logging.get_logger(__name__)

__all__ = ["FalconConfig"]


class FalconConfig(PretrainedConfig):
    r"""
    Falcon config
    """

    model_type = "falcon"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65024,
        hidden_size=4544,
        num_hidden_layers=32,
        num_attention_heads=71,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_kv_heads=None,
        alibi=False,
        new_decoder_architecture=False,
        multi_query=True,
        parallel_attn=True,
        bias=False,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rope_scaling=None,
        bos_token_id=11,
        eos_token_id=11,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.num_kv_heads = (
            num_attention_heads if num_kv_heads is None else num_kv_heads
        )
        self.alibi = alibi
        self.new_decoder_architecture = new_decoder_architecture
        self.multi_query = multi_query  # Ignored when new_decoder_architecture is True
        self.parallel_attn = parallel_attn
        self.bias = bias
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def head_dim(self):
        """
        Gets the dimension of each attention head.

        Returns:
            int: The dimension of each attention head."""

        return self.hidden_size // self.num_attention_heads

    @property
    def rotary(self):
        """
        Checks if the rotary property is enabled.

        Returns:
            bool: True if the rotary property is enabled, False otherwise."""

        return not self.alibi

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if self.rotary:
            raise ValueError("`rope_scaling` is not supported when `alibi` is `True`.")

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if (
            rope_scaling_factor is None
            or not isinstance(rope_scaling_factor, float)
            or rope_scaling_factor <= 1.0
        ):
            raise ValueError(
                f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}"
            )
