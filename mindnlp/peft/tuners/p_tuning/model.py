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

# Based on https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/prompt_encoder.py
# with some refactor
"""p-tuning model"""
import warnings

from mindnlp.core import nn

from .config import PromptEncoderConfig, PromptEncoderReparameterizationType


class PromptEncoder(nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.

    Args:
        config ([`PromptEncoderConfig`]): The configuration of the prompt encoder.

    Example:

    ```py
    >>> from peft import PromptEncoder, PromptEncoderConfig

    >>> config = PromptEncoderConfig(
    ...     peft_type="P_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     encoder_reparameterization_type="MLP",
    ...     encoder_hidden_size=768,
    ... )

    >>> prompt_encoder = PromptEncoder(config)
    ```

    **Attributes**:
        - **embedding** (`nn.Embedding`) -- The embedding layer of the prompt encoder.
        - **mlp_head** (`nn.Sequential`) -- The MLP head of the prompt encoder if `inference_mode=False`.
        - **lstm_head** (`nn.LSTM`) -- The LSTM head of the prompt encoder if `inference_mode=False` and
        `encoder_reparameterization_type="LSTM"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
        prompt encoder.
        - **encoder_type** (Union[[`PromptEncoderReparameterizationType`], `str`]): The encoder type of the prompt
          encoder.


    Input shape: (`batch_size`, `total_virtual_tokens`)

    Output shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """
    def __init__(self, config):
        """
        Initializes a PromptEncoder instance.
        
        Args:
            self (PromptEncoder): The instance of the PromptEncoder class.
            config (PromptEncoderConfig): An object containing configuration parameters for the PromptEncoder.
                The configuration should include the following attributes:
                    - token_dim (int): The dimensionality of the token embeddings.
                    - encoder_hidden_size (int): The size of the hidden layer in the encoder.
                    - num_virtual_tokens (int): The number of virtual tokens.
                    - num_transformer_submodules (int): The number of transformer submodules.
                    - encoder_reparameterization_type (PromptEncoderReparameterizationType): The type of encoder reparameterization.
                    - encoder_dropout (float): The dropout rate for the encoder.
                    - encoder_num_layers (int): The number of layers in the encoder.
                    - inference_mode (bool): Flag indicating whether the model is in inference mode.
        
        Returns:
            None. This method initializes the PromptEncoder instance with the provided configuration settings.
        
        Raises:
            ValueError: If the encoder type specified in the configuration is not recognized. Accepted types are MLP or LSTM.
            Warning: If the specified number of encoder layers is different from the default value when using the MLP encoder type.
        """
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.encoder_type = config.encoder_reparameterization_type

        # embedding
        self.embedding = nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.inference_mode:
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                lstm_dropout = config.encoder_dropout
                num_layers = config.encoder_num_layers
                # LSTM
                self.lstm_head = nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )

                self.mlp_head = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size * 2, self.output_size),
                )

            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                encoder_num_layers_default = PromptEncoderConfig.encoder_num_layers
                if config.encoder_num_layers != encoder_num_layers_default:
                    warnings.warn(
                        f"for {self.encoder_type.value}, the argument `encoder_num_layers` is ignored. "
                        f"Exactly {encoder_num_layers_default} MLP layers are used."
                    )
                layers = [
                    nn.Linear(self.input_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, self.output_size),
                ]
                self.mlp_head = nn.Sequential(*layers)

            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    def forward(self, indices):
        """
        Forward method in the PromptEncoder class.
        
        This method takes in two parameters, self and indices, and returns None.
        
        Args:
            self: An instance of the PromptEncoder class.
            indices (Tensor): A tensor containing the indices used for embedding lookup. The shape of the tensor should be (batch_size, sequence_length), where batch_size is the number of input sequences, and
sequence_length is the length of each input sequence. Each element in the tensor represents the index of a word in the vocabulary.
            
        Returns:
            output_embeds (Tensor): A tensor containing the output embeddings. The shape of the tensor depends on the encoder type. If the encoder_type is PromptEncoderReparameterizationType.LSTM, the shape
will be (batch_size, sequence_length, embedding_size), where embedding_size is the size of the embedding vector. If the encoder_type is PromptEncoderReparameterizationType.MLP, the shape will be (batch_size,
sequence_length, output_size), where output_size is the size of the output vector.
        
        Raises:
            ValueError: If the encoder_type is not recognized. Please use either PromptEncoderReparameterizationType.MLP or PromptEncoderReparameterizationType.LSTM.
        
        """
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
