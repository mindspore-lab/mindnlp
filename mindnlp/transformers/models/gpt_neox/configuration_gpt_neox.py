# coding=utf-8
# Copyright 2024 Huawei Technologies Co., Ltd
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
""" GPTNeoX model configuration"""

from mindnlp.utils import logging
from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-neox-20b": "https://hf-mirror.com/EleutherAI/gpt-neox-20b/resolve/main/config.json",
    # See all GPTNeoX models at https://hf-mirror.com/models?filter=gpt_neox
}


class GPTNeoXConfig(PretrainedConfig):
    r"""
    GPTNeoX config
    """
    model_type = "gpt_neox"

    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        rope_scaling=None,
        attention_bias=True,
        **kwargs,
    ):
        """
        Initialize a new GPTNeoXConfig instance.
        
        Args:
            vocab_size (int, optional): The size of the vocabulary. Defaults to 50432.
            hidden_size (int, optional): The hidden size of the model. Defaults to 6144.
            num_hidden_layers (int, optional): The number of hidden layers in the model. Defaults to 44.
            num_attention_heads (int, optional): The number of attention heads. Defaults to 64.
            intermediate_size (int, optional): The size of the intermediate layer in the model. Defaults to 24576.
            hidden_act (str, optional): The activation function for the hidden layers. Defaults to 'gelu'.
            rotary_pct (float, optional): The percentage of rotary embeddings. Defaults to 0.25.
            rotary_emb_base (int, optional): The base value for rotary embeddings. Defaults to 10000.
            attention_dropout (float, optional): The dropout rate for attention layers. Defaults to 0.0.
            hidden_dropout (float, optional): The dropout rate for hidden layers. Defaults to 0.0.
            classifier_dropout (float, optional): The dropout rate for the classifier layer. Defaults to 0.1.
            max_position_embeddings (int, optional): The maximum position embeddings. Defaults to 2048.
            initializer_range (float, optional): The range for parameter initialization. Defaults to 0.02.
            layer_norm_eps (float, optional): The epsilon value for layer normalization. Defaults to 1e-05.
            use_cache (bool, optional): Whether to use cache for decoding. Defaults to True.
            bos_token_id (int, optional): The beginning of sequence token id. Defaults to 0.
            eos_token_id (int, optional): The end of sequence token id. Defaults to 2.
            tie_word_embeddings (bool, optional): Whether to tie word embeddings. Defaults to False.
            use_parallel_residual (bool, optional): Whether to use parallel residual connections. Defaults to True.
            rope_scaling (NoneType, optional): The scaling factor for the relative position encoding. Defaults to None.
            attention_bias (bool, optional): Whether to use attention bias. Defaults to True.
        
        Returns:
            None.
        
        Raises:
            ValueError: If the hidden size is not divisible by the number of attention heads.
        """
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_parallel_residual = use_parallel_residual
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self._rope_scaling_validation()

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them!"
            )

    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

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
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


__all__ = [
    "GPTNeoXConfig",
]
