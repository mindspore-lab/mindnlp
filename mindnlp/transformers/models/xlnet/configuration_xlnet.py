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
""" XLNet configuration"""

import warnings

from ...configuration_utils import PretrainedConfig
from ....utils import logging

logger = logging.get_logger(__name__)

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xlnet/xlnet-base-cased": "https://huggingface.co/xlnet/xlnet-base-cased/resolve/main/config.json",
    "xlnet/xlnet-large-cased": "https://huggingface.co/xlnet/xlnet-large-cased/resolve/main/config.json",
}


class XLNetConfig(PretrainedConfig):
    """
    Configuration for XLNet
    """
    model_type = "xlnet"
    keys_to_ignore_at_inference = ["mems"]
    attribute_map = {
        "n_token": "vocab_size",  # Backward compatibility
        "hidden_size": "d_model",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size=32000,
            d_model=1024,
            n_layer=24,
            n_head=16,
            d_inner=4096,
            ff_activation="gelu",
            untie_r=True,
            attn_type="bi",
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            dropout=0.1,
            mem_len=512,
            reuse_len=None,
            use_mems_eval=True,
            use_mems_train=False,
            bi_data=False,
            clamp_len=-1,
            same_length=False,
            summary_type="last",
            summary_use_proj=True,
            summary_activation="tanh",
            summary_last_dropout=0.1,
            start_n_top=5,
            end_n_top=5,
            pad_token_id=5,
            bos_token_id=1,
            eos_token_id=2,
            **kwargs,
    ):
        """Constructs XLNetConfig."""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        if d_model % n_head != 0:
            raise ValueError(f"'d_model % n_head' ({d_model % n_head}) should be equal to 0")
        if "d_head" in kwargs:
            if kwargs["d_head"] != d_model // n_head:
                raise ValueError(
                    f"`d_head` ({kwargs['d_head']}) should be equal to `d_model // n_head` ({d_model // n_head})"
                )
        self.d_head = d_model // n_head
        self.ff_activation = ff_activation
        self.d_inner = d_inner
        self.untie_r = untie_r
        self.attn_type = attn_type

        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.dropout = dropout
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top

        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated, use `use_mems_eval`"
                " instead.",
                FutureWarning,
            )
            use_mems_eval = kwargs["use_cache"]

        self.use_mems_eval = use_mems_eval
        self.use_mems_train = use_mems_train
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

    @property
    def max_position_embeddings(self):
        """
        This method returns the maximum position embeddings for the XLNet model.
        
        Args:
            self (XLNetConfig): The instance of the XLNetConfig class.
            
        Returns:
            None: This method does not return any specific value, as it only logs a message and returns -1.
        
        Raises:
            None
        """
        logger.info(f"The model {self.model_type} is one of the few models that has no sequence length limit.")
        return -1

    @max_position_embeddings.setter
    def max_position_embeddings(self, value):
        """
        Sets the maximum position embeddings for the XLNetConfig class.
        
        Args:
            self (XLNetConfig): An instance of the XLNetConfig class.
            value: The desired value for the maximum position embeddings. It should be an integer.
            
        Returns:
            None.
            
        Raises:
            NotImplementedError: This exception is raised when trying to set the maximum position embeddings for
                the XLNetConfig class. Since the model type is one of the few models that has no sequence length
                limit, setting the maximum position embeddings is not allowed.
        
        Note:
            The model type should be specified before using this method.
        """
        # Message copied from Transformer-XL documentation
        raise NotImplementedError(
            f"The model {self.model_type} is one of the few models that has no sequence length limit."
        )


__all__ = ['XLNetConfig']
