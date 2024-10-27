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
"""prompt tuning model"""
import math

import mindspore
from mindnlp.core.nn import Parameter
from mindnlp.core import nn
from .config import PromptTuningInit

class PromptEmbedding(nn.Module):
    """
    The model to encode virtual tokens into prompt embeddings.

    Args:
        config ([`PromptTuningConfig`]): The configuration of the prompt embedding.
        word_embeddings (`nn.Module`): The word embeddings of the base transformer model.

    **Attributes**:
        - **embedding** (`nn.Embedding`) -- The embedding layer of the prompt embedding.

    Example:

    ```py
    >>> from peft import PromptEmbedding, PromptTuningConfig

    >>> config = PromptTuningConfig(
    ...     peft_type="PROMPT_TUNING",
    ...     task_type="SEQ_2_SEQ_LM",
    ...     num_virtual_tokens=20,
    ...     token_dim=768,
    ...     num_transformer_submodules=1,
    ...     num_attention_heads=12,
    ...     num_layers=12,
    ...     prompt_tuning_init="TEXT",
    ...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
    ...     tokenizer_name_or_path="t5-base",
    ... )

    >>> # t5_model.shared is the word embeddings of the base model
    >>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
    ```

    Input Shape: (`batch_size`, `total_virtual_tokens`)

    Output Shape: (`batch_size`, `total_virtual_tokens`, `token_dim`)
    """
    def __init__(self, config, word_embeddings):
        r"""
        Initialize the PromptEmbedding class.
        
        Args:
            self: Reference to the current instance of the class.
            config (object): Configuration object containing various settings.
                - num_virtual_tokens (int): Number of virtual tokens.
                - num_transformer_subcells (int): Number of transformer subcells.
                - token_dim (int): Dimensionality of the token embeddings.
                - prompt_tuning_init (Enum): Specifies the type of prompt tuning initialization.
                - inference_mode (bool): Indicates if the model is in inference mode.
                - tokenizer_kwargs (dict, optional): Additional keyword arguments for the tokenizer.
                - tokenizer_name_or_path (str): Name or path of the pretrained tokenizer.
                - prompt_tuning_init_text (str): Text used for prompt tuning initialization.
            word_embeddings (object): Word embeddings for initializing the embedding layer.
        
        Returns:
            None. The method initializes the embedding layer with the provided word embeddings.
        
        Raises:
            ImportError: If the transformers module cannot be imported.
            ValueError: If the number of text tokens exceeds the total virtual tokens.
            TypeError: If the word embedding weights cannot be converted to float32.
        """
        super().__init__()

        total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.embedding = nn.Embedding(total_virtual_tokens, config.token_dim)
        if config.prompt_tuning_init == PromptTuningInit.TEXT and not config.inference_mode:
            from ....transformers import AutoTokenizer

            tokenizer_kwargs = config.tokenizer_kwargs or {}
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, **tokenizer_kwargs)
            init_text = config.prompt_tuning_init_text
            init_token_ids = tokenizer(init_text)["input_ids"]
            # Trim or iterate until num_text_tokens matches total_virtual_tokens
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = mindspore.tensor(init_token_ids)
            word_embedding_weights = word_embeddings(init_token_ids).copy()
            word_embedding_weights = word_embedding_weights.to(mindspore.float32)
            self.embedding.weight = Parameter(word_embedding_weights)

    def forward(self, indices):
        r"""
        Construct the prompt embeddings based on the given indices.
        
        Args:
            self (PromptEmbedding): An instance of the PromptEmbedding class.
            indices (int): The indices used to retrieve the prompt embeddings.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            None: This method does not raise any exceptions.
        """
        # Just get embeddings
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings
