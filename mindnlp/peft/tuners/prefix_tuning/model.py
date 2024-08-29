# Copyright 2023-present the HuggingFace Inc. team.
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

# Based on https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
# with some refactor
"""prefix tuning model"""
import mindspore
from mindnlp.core import nn

class PrefixEncoder(nn.Module):
    r"""
    The `mindspore.nn` model to encode the prefix.

    Args:
        config ([`PrefixTuningConfig`]): The configuration of the prefix encoder.

    Example:

    ```py
    >>> from peft import PrefixEncoder, PrefixTuningConfig

    >>> config = PrefixTuningConfig(
    ...     peft_type="PREFIX_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_hidden_size=768,
    ... )
    >>> prefix_encoder = PrefixEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`mindspore.nn.Embedding`) -- The embedding layer of the prefix encoder.
        - **transform** (`mindspore.nn.Sequential`) -- The two-layer MLP to transform the prefix embeddings if
          `prefix_projection` is `True`.
        - **prefix_projection** (`bool`) -- Whether to project the prefix embeddings.

    Input shape: (`batch_size`, `num_virtual_tokens`)

    Output shape: (`batch_size`, `num_virtual_tokens`, `2*layers*hidden`)
    """
    def __init__(self, config):
        """
        Initializes the PrefixEncoder class.
        
        Args:
            self: The object instance.
            config (object): A configuration object containing the following attributes:
                - prefix_projection (bool): Indicates whether prefix projection should be applied.
                - token_dim (int): The dimension of the token embedding.
                - num_layers (int): The number of layers in the encoder.
                - encoder_hidden_size (int): The size of the hidden state in the encoder.
                - num_virtual_tokens (int): The number of virtual tokens.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If the prefix_projection attribute is True and the inference_mode attribute in the config object is not set.
        """
        super().__init__()
        self.prefix_projection = config.prefix_projection
        token_dim = config.token_dim
        num_layers = config.num_layers
        encoder_hidden_size = config.encoder_hidden_size
        num_virtual_tokens = config.num_virtual_tokens
        if self.prefix_projection and not config.inference_mode:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Embedding(num_virtual_tokens, token_dim)
            self.transform = nn.Sequential(
                nn.Linear(token_dim, encoder_hidden_size),
                nn.Tanh(),
                nn.Linear(encoder_hidden_size, num_layers * 2 * token_dim),
            )
        else:
            self.embedding = nn.Embedding(num_virtual_tokens, num_layers * 2 * token_dim)

    def forward(self, prefix: mindspore.Tensor):
        """
        This method forwards the past key values based on the provided prefix for the PrefixEncoder.
        
        Args:
            self (PrefixEncoder): The instance of the PrefixEncoder class.
            prefix (mindspore.Tensor): The input prefix tensor used for forwarding past key values.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - TypeError: If the prefix is not of type mindspore.Tensor.
            - ValueError: If the prefix projection is enabled and the prefix_tokens cannot be obtained or transformed.
            - RuntimeError: If there is an issue with the embedding or transformation process.
        """
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.transform(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
