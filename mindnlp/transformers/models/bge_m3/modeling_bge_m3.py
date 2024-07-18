# Copyright 2024 Huawei Technologies Co., Ltd
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
Bge M3 Config.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import mindspore
from mindspore import nn, ops

from ...modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, ModelOutput
from ..xlm_roberta import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)

from .configuration_bge_m3 import BgeM3Config


@dataclass
class BgeM3ModelOutput(ModelOutput):

    """
    Represents the output of a BGE M3 model. 
    
    This class inherits from the ModelOutput class and provides specific functionality for handling outputs
    from BGE M3 models.
    """
    last_hidden_state: mindspore.Tensor = None
    pooler_output: mindspore.Tensor = None
    dense_output: mindspore.Tensor = None
    colbert_output: Optional[List[mindspore.Tensor]] = None
    sparse_output: Optional[Dict[int, float]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor]] = None
    past_key_values: Optional[Tuple[Tuple[mindspore.Tensor]]] = None
    attentions: Optional[Tuple[mindspore.Tensor]] = None
    cross_attentions: Optional[Tuple[mindspore.Tensor]] = None


class BgeM3Model(XLMRobertaPreTrainedModel):

    """
    The BgeM3Model class represents a model that extends XLMRobertaPreTrainedModel.
    It includes methods for dense embedding, sparse embedding, Colbert embedding, and processing token weights
    and Colbert vectors.
    The construct method processes input tensors to generate various outputs including last hidden state, dense output,
    pooler output, Colbert output, sparse output, hidden states, past key values, attentions, and cross attentions.
    """
    config_class = BgeM3Config

    def __init__(self, config: BgeM3Config):
        """
            Initializes a new instance of the BgeM3Model class.

            Args:
                self: The current BgeM3Model instance.
                config (BgeM3Config): The configuration object for BgeM3Model.

            Returns:
                None

            Raises:
                None
            """
        super().__init__(config)
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.colbert_linear = nn.Dense(
            config.hidden_size,
            config.hidden_size if config.colbert_dim is None else config.colbert_dim,
        )
        self.sparse_linear = nn.Dense(config.hidden_size, 1)
        self.sentence_pooling_method = config.sentence_pooling_method

        self.init_weights()

    # Copied from FlagEmbedding
    def dense_embedding(self, hidden_state, mask):
        """
        This method calculates the dense embedding based on the provided hidden state and mask,
        using the specified sentence pooling method.

        Args:
            self (object): The instance of the BgeM3Model class.
            hidden_state (tensor): The hidden state tensor representing the input sequence.
            mask (tensor): The mask tensor indicating the presence of valid elements in the input sequence.
                Its shape should be compatible with hidden_state.

        Returns:
            None: This method does not return a value, as the dense embedding is directly computed and returned.

        Raises:
            ValueError: If the sentence pooling method specified is not supported or recognized.
            RuntimeError: If there are issues with the tensor operations or calculations within the method.
        """
        if self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "mean":
            s = ops.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    # Copied from FlagEmbedding
    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = False):
        """
        Sparse Embedding

        This method computes the sparse embedding for a given hidden state and input IDs.

        Args:
            self (BgeM3Model): The instance of the BgeM3Model class.
            hidden_state: The hidden state tensor.
            input_ids: The input IDs tensor.
            return_embedding (bool, optional): Whether to return the sparse embedding or token weights.
                Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        token_weights = ops.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        sparse_embedding = ops.zeros(
            (input_ids.shape[0],
            input_ids.shape[1],
            self.config.vocab_size),
            dtype=token_weights.dtype,
        )
        sparse_embedding = ops.scatter(sparse_embedding, axis=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = self.config.unused_tokens
        sparse_embedding = ops.max(sparse_embedding, axis=1)[0]
        sparse_embedding[:, unused_tokens] *= 0.0
        return sparse_embedding

    # Copied from FlagEmbedding
    def colbert_embedding(self, last_hidden_state, mask):
        """
        Embeds the last hidden state of the BgeM3Model using the Colbert method.

        Args:
            self (BgeM3Model): The instance of the BgeM3Model class.
            last_hidden_state (torch.Tensor): The last hidden state of the model.
                Shape: (batch_size, sequence_length, hidden_size)
            mask (torch.Tensor): The mask specifying the valid positions in the last_hidden_state tensor.
                Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: The embedded Colbert vectors.
                Shape: (batch_size, sequence_length-1, hidden_size)

        Raises:
            None
        """
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    # Modified from FlagEmbedding
    def _process_token_weights(self, token_weights, input_ids, mask):
        """
        Process the token weights for the BgeM3Model.

        Args:
            self (BgeM3Model): An instance of the BgeM3Model class.
            token_weights (Tensor): A tensor containing the weights of each token.
            input_ids (Tensor): A tensor containing the input IDs.
            mask (Tensor): A tensor containing the mask.

        Returns:
            list[defaultdict]:
                A list of dictionaries, where each dictionary contains the maximum weight for each unique ID.

        Raises:
            None.

        This method processes the given token weights by removing unused tokens and filtering out invalid indices.
        It then computes the maximum weight for each unique ID and stores the results in a list of dictionaries.
        The resulting list is returned as the output of this method.
        """
        token_weights = token_weights.squeeze(-1)
        # conver to dict
        all_result = []
        unused_tokens = self.config.unused_tokens
        unused_tokens = mindspore.tensor(unused_tokens)

        # Get valid matrix
        valid_indices = ~mindspore.numpy.isin(input_ids, unused_tokens)
        # w>0
        valid_indices = (valid_indices & (token_weights > 0)).bool()
        valid_indices = (valid_indices & mask).bool()

        for i, valid in enumerate(valid_indices):
            result = defaultdict(int)

            # Get valid weight and ids
            valid_weights = token_weights[i][valid]
            valid_ids = input_ids[i][valid]

            # Get unique token
            unique_ids, inverse_indices = ops.unique(valid_ids)

            # Get max weight for each token
            for i in range(unique_ids.shape[0]):
                id_mask = inverse_indices == i
                result[str(unique_ids[i].item())] = valid_weights[id_mask].max().item()

            all_result.append(result)

        return all_result

    # Copied from FlagEmbedding
    def _process_colbert_vecs(self, colbert_vecs, tokens_num) -> List[mindspore.Tensor]:
        '''
        This method processes the Colbert vectors to extract a subset of vectors based on the tokens number.

        Args:
            self (BgeM3Model): The instance of the BgeM3Model class.
            colbert_vecs (Union[mindspore.Tensor, List[mindspore.Tensor]]): The Colbert vectors to be processed.
            tokens_num (List[int]): The list containing the number of tokens for each vector in colbert_vecs.

        Returns:
            List[mindspore.Tensor]: A list of processed vectors.

        Raises:
            ValueError: If the length of colbert_vecs and tokens_num does not match.
            IndexError: If the tokens_num contains an index that is out of range for colbert_vecs.
        '''
        # delte the vectors of padding tokens
        vecs = []
        for i in range(len(tokens_num)):
            vecs.append(colbert_vecs[i, : tokens_num[i] - 1])
        return vecs

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def construct(
        self,
        input_ids: Optional[mindspore.Tensor] = None,
        attention_mask: Optional[mindspore.Tensor] = None,
        token_type_ids: Optional[mindspore.Tensor] = None,
        position_ids: Optional[mindspore.Tensor] = None,
        head_mask: Optional[mindspore.Tensor] = None,
        inputs_embeds: Optional[mindspore.Tensor] = None,
        encoder_hidden_states: Optional[mindspore.Tensor] = None,
        encoder_attention_mask: Optional[mindspore.Tensor] = None,
        past_key_values: Optional[List[mindspore.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[mindspore.Tensor], BgeM3ModelOutput]:
        """
        Constructs the BgeM3Model.

        Args:
            self: The instance of the class.
            input_ids (Optional[mindspore.Tensor]):
                The input tensor of shape (batch_size, sequence_length) containing the input IDs.
            attention_mask (Optional[mindspore.Tensor]):
                The attention mask tensor of shape (batch_size, sequence_length) containing attention masks
                for the input IDs.
            token_type_ids (Optional[mindspore.Tensor]):
                The token type IDs tensor of shape (batch_size, sequence_length) containing the token type IDs
                for the input IDs.
            position_ids (Optional[mindspore.Tensor]):
                The position IDs tensor of shape (batch_size, sequence_length) containing the position IDs
                for the input IDs.
            head_mask (Optional[mindspore.Tensor]):
                The head mask tensor of shape (num_heads,) or (num_layers, num_heads) containing the head mask
                for the transformer encoder.
            inputs_embeds (Optional[mindspore.Tensor]):
                The input embeddings tensor of shape (batch_size, sequence_length, hidden_size) containing
                the embeddings for the input IDs.
            encoder_hidden_states (Optional[mindspore.Tensor]):
                The encoder hidden states tensor of shape (batch_size, encoder_sequence_length, hidden_size)
                containing the hidden states of the encoder.
            encoder_attention_mask (Optional[mindspore.Tensor]):
                The encoder attention mask tensor of shape (batch_size, encoder_sequence_length) containing
                attention masks for the encoder hidden states.
            past_key_values (Optional[List[mindspore.Tensor]]):
                The list of past key value tensors of shape (2, batch_size, num_heads, sequence_length,
                hidden_size//num_heads) containing the past key value states for the transformer decoder.
            use_cache (Optional[bool]): Whether to use cache for the transformer decoder.
            output_attentions (Optional[bool]): Whether to output attentions.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary instead of a tuple.

        Returns:
            Union[Tuple[mindspore.Tensor], BgeM3ModelOutput]:
                If `return_dict` is set to False, returns a tuple containing the following elements:

                - last_hidden_state (mindspore.Tensor):
                The last hidden state tensor of shape (batch_size, sequence_length, hidden_size)
                containing the last hidden state of the transformer.
                - pooler_output (mindspore.Tensor):
                The pooler output tensor of shape (batch_size, hidden_size) containing the pooler output of
                the transformer.
                - dense_output (mindspore.Tensor):
                The dense embedding output tensor of shape (batch_size, sequence_length, dense_size) containing
                the dense embeddings.
                - colbert_output (mindspore.Tensor):
                The Colbert embedding output tensor of shape (batch_size, sequence_length, colbert_size) containing
                the Colbert embeddings.
                - sparse_output (mindspore.Tensor):
                The sparse embedding output tensor of shape (batch_size, sequence_length, sparse_size) containing
                the sparse embeddings.
                - hidden_states (Tuple[mindspore.Tensor]):
                The hidden states tensor of shape (num_layers, batch_size, sequence_length, hidden_size) containing
                the hidden states of the transformer.
                - past_key_values (Tuple[mindspore.Tensor]):
                The past key value tensors of shape (2, batch_size, num_heads, sequence_length, hidden_size//num_heads)
                containing the past key value states for the transformer decoder.
                - attentions (Tuple[mindspore.Tensor]):
                The attentions tensors of shape (num_layers, batch_size, num_heads, sequence_length, sequence_length)
                containing the attentions of the transformer.
                - cross_attentions (Tuple[mindspore.Tensor]):
                The cross attentions tensors of shape (num_layers, batch_size, num_heads, sequence_length, encoder_sequence_length)
                containing the cross attentions of the transformer.

            BgeM3ModelOutput:
                If `return_dict` is set to True, returns an instance of the BgeM3ModelOutput class containing the
                following elements:

                - last_hidden_state (mindspore.Tensor):
                The last hidden state tensor of shape (batch_size, sequence_length, hidden_size)
                containing the last hidden state of the transformer.
                - dense_output (mindspore.Tensor):
                The dense embedding output tensor of shape (batch_size, sequence_length, dense_size)
                containing the dense embeddings.
                - pooler_output (mindspore.Tensor):
                The pooler output tensor of shape (batch_size, hidden_size)
                containing the pooler output of the transformer.
                - colbert_output (mindspore.Tensor):
                The Colbert embedding output tensor of shape (batch_size, sequence_length, colbert_size)
                containing the Colbert embeddings.
                - sparse_output (mindspore.Tensor):
                The sparse embedding output tensor of shape (batch_size, sequence_length, sparse_size)
                containing the sparse embeddings.
                - hidden_states (Tuple[mindspore.Tensor]):
                The hidden states tensor of shape (num_layers, batch_size, sequence_length, hidden_size)
                containing the hidden states of the transformer.
                - past_key_values (Tuple[mindspore.Tensor]):
                The past key value tensors of shape (2, batch_size, num_heads, sequence_length, hidden_size//num_heads)
                containing the past key value states for the transformer decoder.
                - attentions (Tuple[mindspore.Tensor]):
                The attentions tensors of shape (num_layers, batch_size, num_heads, sequence_length, sequence_length)
                containing the attentions of the transformer.
                - cross_attentions (Tuple[mindspore.Tensor]):
                The cross attentions tensors of shape (num_layers, batch_size, num_heads, sequence_length, encoder_sequence_length)
                containing the cross attentions of the transformer.
        
        Raises:
            None.
        """
        roberta_output: BaseModelOutputWithPoolingAndCrossAttentions = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        last_hidden_state = roberta_output.last_hidden_state
        dense_output = self.dense_embedding(last_hidden_state, attention_mask)

        tokens_num = attention_mask.sum(axis=1)
        colbert_output = self.colbert_embedding(last_hidden_state, attention_mask)
        colbert_output = self._process_colbert_vecs(colbert_output, tokens_num)

        sparse_output = self.sparse_embedding(last_hidden_state, input_ids)
        sparse_output = self._process_token_weights(sparse_output, input_ids, attention_mask)

        if not return_dict:
            return (
                last_hidden_state,
                roberta_output.pooler_output,
                dense_output,
                colbert_output,
                sparse_output,
                roberta_output.hidden_states,
                roberta_output.past_key_values,
                roberta_output.attentions,
                roberta_output.cross_attentions,
            )

        return BgeM3ModelOutput(
            last_hidden_state=last_hidden_state,
            dense_output=dense_output,
            pooler_output=roberta_output.pooler_output,
            colbert_output=colbert_output,
            sparse_output=sparse_output,
            hidden_states=roberta_output.hidden_states,
            past_key_values=roberta_output.past_key_values,
            attentions=roberta_output.attentions,
            cross_attentions=roberta_output.cross_attentions,
        )

__all__ = ['BgeM3Model']
