# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

"""MindNLP gpt model"""
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.common.initializer import initializer, Normal

from mindnlp.core import nn, ops
from mindnlp.core.nn import functional as F
from .configuration_openai import OpenAIGPTConfig
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...ms_utils import Conv1D, prune_conv1d_layer, find_pruneable_heads_and_indices
from ...activations import ACT2FN


class MLP(nn.Module):
    r"""
    GPT MLP
	"""
    def __init__(self, n_state, config):
        """
        Initializes an instance of the MLP class.
        
        Args:
            self (object): The instance of the MLP class.
            n_state (int): The number of states for the neural network model.
            config (object): An object containing configuration settings for the MLP.
            
        Returns:
            None.
        
        Raises:
            None.
        """
        super().__init__()
        n_embd = config.n_embd
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = ACT2FN[config.afn]
        self.dropout = nn.Dropout(p=config.resid_pdrop)

    def forward(self, x):
        """
        Constructs the output of the MLP (Multi-Layer Perceptron) for a given input.
        
        Args:
            self (MLP): An instance of the MLP class.
            x: The input tensor of shape (batch_size, input_size) representing the input data.
               It should be a 2-dimensional tensor where the first dimension represents the batch size
               and the second dimension represents the input size of the MLP.
        
        Returns:
            None
        
        Raises:
            None
        
        Description:
            This method takes an input tensor 'x' and applies the forward pass of the MLP.
            It performs the following steps:

            1. Applies the activation function to the linearly transformed input tensor using the 'c_fc' layer.
            2. Passes the result through the 'c_proj' layer to obtain the final output.
            3. Applies dropout to the output tensor to prevent overfitting.

            The forward pass of the MLP is defined as:
                ```python
                >>> h = self.act(self.c_fc(x))
                >>> h2 = self.c_proj(h)
                >>> return self.dropout(h2)
                ```

        Example:
            ```python
            >>> mlp = MLP()
            >>> input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> mlp.forward(input_tensor)
            ```
        """
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Attention(nn.Module):
    r"""
    GPT Attention
    """
    def __init__(self, nx, n_positions, config, scale=False):
        """
        Initializes an instance of the Attention class.

        Args:
            self: The instance of the class.
            nx (int): The number of input units.
            n_positions (int): The number of positions to attend.
            config: The configuration object.
            scale (bool): Indicates whether to scale the attention scores. Defaults to False.

        Returns:
            None

        Raises:
            ValueError: If `nx` is not divisible by `config.n_head`.

        """
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implementation]
        if n_state % config.n_head != 0:
            raise ValueError(f"Attention n_state shape: {n_state} must be divisible by config.n_head {config.n_head}")

        self.bias = Tensor(np.tril(np.ones((n_positions, n_positions))), mindspore.float32).view(1, 1, n_positions, n_positions)
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_attn = Conv1D(n_state * 3, n_state)
        self.c_proj = Conv1D(n_state, n_state)
        self.attn_dropout = nn.Dropout(p=config.attn_pdrop)
        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)
        self.pruned_heads = set()

        self.output_attentions = config.output_attentions

    def prune_heads(self, heads):
        """
        Prunes heads of the model.
        """
        if len(heads) == 0:
            return
        head_size = self.split_size//self.n_head
        heads, index = find_pruneable_heads_and_indices(heads, self.n_head, head_size, self.pruned_heads)
        index_attn = ops.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        """
        Method _attn in the Attention class.

        Args:
            self: Attention object. Instance of the Attention class.
            q: torch.Tensor. Query tensor.
            k: torch.Tensor. Key tensor.
            v: torch.Tensor. Value tensor.
            attention_mask: torch.Tensor, optional. Mask tensor for attention scores.
            head_mask: torch.Tensor, optional. Mask tensor for heads.

        Returns:
            None: The method does not explicitly return any value, but it modifies the internal state of the Attention
                object.

        Raises:
            None specified.
        """
        w = ops.matmul(q, k)
        if self.scale:
            w = w / ops.sqrt(ops.scalar_to_tensor(v.shape[-1]))
        b = self.bias[:, :, : w.shape[-2], : w.shape[-1]]
        w = w * b + -1e9 * (1 - b)

        if attention_mask is not None:
            w = w + attention_mask

        w = ops.softmax(w)
        w = self.attn_dropout(w)

        if head_mask is not None:
            w = w * head_mask

        outputs = (self.matmul(w, v),)
        if self.output_attentions:
            outputs += (w,)
        return outputs

    def merge_heads(self, x):
        """merge heads"""
        x = x.transpose(0, 2, 1, 3)
        new_x_shape = x.shape[:-2] + (x.shape[-2] * x.shape[-1],)
        return x.view(new_x_shape)

    def split_heads(self, x, k=False):
        """split heads"""
        new_x_shape = x.shape[:-1] + (self.n_head, x.shape[-1] // self.n_head)
        x = x.view(new_x_shape)
        if k:
            return x.transpose(0, 2, 3, 1)
        return x.transpose(0, 2, 1, 3)

    def forward(self, x, attention_mask=None, head_mask=None):
        """
        Constructs the attention mechanism in the Attention class.

        Args:
            self (object): The instance of the class.
            x (tensor): The input tensor to the attention mechanism.
            attention_mask (tensor, optional): An optional attention mask tensor.
            head_mask (tensor, optional): An optional head mask tensor.

        Returns:
            tuple: A tuple containing the output tensors of the attention mechanism.

        Raises:
            ValueError: If the dimensions of the input tensors are incompatible.
            TypeError: If the input tensors are not of the expected type.
            RuntimeError: If an error occurs during the attention mechanism computation.
        """
        x = self.c_attn(x)
        query, key, value = ops.split(x, self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = (a,) + attn_outputs[1:]
        return outputs


class Block(nn.Module):
    r"""
    GPT Block
    """
    def __init__(self, n_positions, config, scale=False):
        """
        Initializes a new instance of the Block class.

        Args:
            self: The instance of the class.
            n_positions (int): The number of positions.
            config: The config object.
            scale (bool, optional): Indicates whether to scale the attention weights. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm((nx,), eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm((nx,), eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None):
        """
        Construct a block by applying attention, normalization, and multi-layer perceptron operations on the input tensor.

        Args:
            self (Block): An instance of the Block class.
            x (torch.Tensor): The input tensor to be processed by the block.
            attention_mask (torch.Tensor, optional): An optional tensor used for masking the attention scores. Default is None.
            head_mask (torch.Tensor, optional): An optional tensor used for masking individual attention heads. Default is None.

        Returns:
            tuple: A tuple containing the processed tensor and any additional outputs from the attention layer.

        Raises:
            None: This method does not raise any exceptions.
        """
        output_attn = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask
        )

        a = output_attn[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = (h,) + output_attn[1:]
        return outputs


class GPTPreTrainedModel(PreTrainedModel):
    """BertPretrainedModel"""
    config_class = OpenAIGPTConfig
    base_model_prefix = 'transformer'

    def _init_weights(self, cell):
        """Initialize the weights"""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                                    cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = initializer(Normal(self.config.initializer_range),
                                                 cell.weight.shape,
                                                 cell.weight.dtype)
            if cell.padding_idx is not None:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(weight)
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))

class GPTModel(GPTPreTrainedModel):
    """
    The bare GPT transformer model outputting raw hidden-states without any specific head on top
    """
    def __init__(self, config):
        """
        Initializes a GPTModel instance with the specified configuration.

        Args:
            self (GPTModel): The GPTModel instance to be initialized.
            config (dict):
                A dictionary containing configuration parameters for the GPTModel.

                - vocab_size (int): The size of the vocabulary.
                - n_embd (int): The dimension of the token embeddings.
                - n_positions (int): The maximum number of positions for positional embeddings.
                - embd_pdrop (float): The dropout probability for embeddings.
                - n_layer (int): The number of layers in the model.
                - output_attentions (bool): Whether to output attention weights.
                - output_hidden_states (bool): Whether to output hidden states.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(p=config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_positions, config, scale=True) for _ in range(config.n_layer)])
        self.position_ids = ops.arange(config.n_positions)

        self.n_layer = self.config.n_layer
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states

    def get_input_embeddings(self):
        """
        return the input embeddings layer
        """
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        """
        set the input embeddings layer
        """
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
    ):
        """
        This method forwards the GPT model based on the provided input parameters.

        Args:
            self: The instance of the class.
            input_ids (optional): A tensor containing the input token IDs. Default is None.
            attention_mask (optional): A tensor specifying the attention mask. Default is None.
            token_type_ids (optional): A tensor containing the token type IDs. Default is None.
            position_ids (optional): A tensor specifying the position IDs. Default is None.
            head_mask (optional): A tensor representing the head mask. Default is None.
            inputs_embeds (optional): A tensor containing the input embeddings. Default is None.

        Returns:
            hidden_states: A tensor representing the final hidden states of the model.
            all_hidden_states: A tuple containing all hidden states from intermediate layers.
            all_attentions: A tuple containing all attention weights from intermediate layers.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified simultaneously,
                if neither input_ids nor inputs_embeds are specified, or for other specific conditions within the method.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, : input_shape[-1]]

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * Tensor(np.finfo(mindspore.dtype_to_nptype(self.dtype)).min,
                                                             self.dtype)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.shape[-1],)

        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(hidden_states, attention_mask, head_mask[i])
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return (hidden_states, all_hidden_states, all_attentions)


class GPTLMHeadModel(GPTPreTrainedModel):
    r"""
    GPT Model transformer with a language modeling head on top
    (linear layer with weights tied to the input embeddings).
    """
    def __init__(self, config):
        """
        Initializes an instance of the GPTLMHeadModel class.

        Args:
            self: The instance of the class.
            config (obj):
                The configuration object containing various settings for the model.

                - This parameter is required.
                - Type: Custom object.
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: None.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.config = config
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def get_output_embeddings(self):
        """
        Returns the embeddings of the obtained output
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Define the embeddings of the output
        """
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
    ):
        """
        Constructs the GPTLMHeadModel.

        Args:
            self (GPTLMHeadModel): The instance of the GPTLMHeadModel class.
            input_ids (torch.Tensor, optional):
                The input tensor of shape (batch_size, sequence_length) containing the input IDs. Defaults to None.
            attention_mask (torch.Tensor, optional):
                The attention mask tensor of shape (batch_size, sequence_length) containing the attention mask.
                Defaults to None.
            token_type_ids (torch.Tensor, optional):
                The token type IDs tensor of shape (batch_size, sequence_length) containing the token type IDs.
                Defaults to None.
            position_ids (torch.Tensor, optional):
                The position IDs tensor of shape (batch_size, sequence_length) containing the position IDs.
                Defaults to None.
            head_mask (torch.Tensor, optional):
                The head mask tensor of shape (num_heads, sequence_length, sequence_length) containing the head mask.
                Defaults to None.
            inputs_embeds (torch.Tensor, optional):
                The input embeddings tensor of shape (batch_size, sequence_length, hidden_size)
                containing the input embeddings. Defaults to None.
            labels (torch.Tensor, optional):
                The labels tensor of shape (batch_size, sequence_length) containing the labels. Defaults to None.

        Returns:
            tuple:
                A tuple containing the following elements:

                - lm_logits (torch.Tensor): The logits tensor of shape (batch_size, sequence_length, vocab_size)
                representing the language model predictions.
                - transformer_outputs (tuple): A tuple containing the transformer outputs.

                    - last_hidden_state (torch.Tensor):
                    The last hidden state of shape (batch_size, sequence_length, hidden_size) from the transformer.
                    - past_key_values (tuple, optional): A tuple containing the past key values.
                    - hidden_states (tuple, optional): A tuple containing the hidden states.
                    - attentions (tuple, optional): A tuple containing the attentions.

        Raises:
            None.
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits,) + transformer_outputs[1:]
        if loss is not None:
            output = (loss,) + output
        return output


class GPTDoubleHeadsModel(GPTPreTrainedModel):
    """
    OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for
    RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the
    input embeddings, the classification head takes as input the input of a specified classification token index in the
    input sequence).
    """
    def __init__(self, config):
        """Initializes a GPTDoubleHeadsModel instance.

        Args:
            self: The GPTDoubleHeadsModel instance.
            config: An instance of the OpenAIGPTConfig class that holds the configuration parameters.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.config = config
        config.num_labels = 1
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the embeddings of the obtained output
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Define the embeddings of the output
        """
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        mc_token_ids = None,
        labels = None,
        mc_labels = None,
    ):
        """
        Constructs the GPTDoubleHeadsModel.

        Args:
            self (GPTDoubleHeadsModel): The instance of the GPTDoubleHeadsModel class.
            input_ids (Tensor, optional): The input tensor of shape ``(batch_size, sequence_length)``.
                It contains the token indices. Defaults to None.
            attention_mask (Tensor, optional): The attention mask tensor of shape ``(batch_size, sequence_length)``.
                It is used to specify which tokens should be attended to and which should not. Defaults to None.
            token_type_ids (Tensor, optional): The token type tensor of shape ``(batch_size, sequence_length)``.
                It is used to indicate the token types (e.g., sentence A and sentence B) in the input sequence.
                Defaults to None.
            position_ids (Tensor, optional): The position indices tensor of shape ``(batch_size, sequence_length)``.
                It is used to specify the position of each token in the input sequence. Defaults to None.
            head_mask (Tensor, optional): The head mask tensor of shape ``(num_layers, num_heads)``.
                It is used to mask certain heads of the attention modules. Defaults to None.
            inputs_embeds (Tensor, optional): The input embeddings tensor of shape
                ``(batch_size, sequence_length, hidden_size)``. It contains the embeddings of the input sequence
                instead of using ``input_ids``. Defaults to None.
            mc_token_ids (Tensor, optional): The multiple-choice token indices tensor of shape
                ``(batch_size, num_choices)``. It contains the token indices for the multiple-choice inputs.
                Defaults to None.
            labels (Tensor, optional): The labels tensor of shape ``(batch_size, sequence_length)``.
                It contains the token indices to predict in the language modeling task. Defaults to None.
            mc_labels (Tensor, optional): The multiple-choice labels tensor of shape ``(batch_size,)``.
                It contains the indices of the correct multiple-choice answers. Defaults to None.

        Returns:
            output (Tuple):
                A tuple containing the following elements:

                - lm_logits (Tensor): The language modeling logits tensor of shape
                ``(batch_size, sequence_length, config.vocab_size)``.
                - mc_logits (Tensor): The multiple-choice logits tensor of shape ``(batch_size, num_choices)``.
                - hidden_states (Tuple): A tuple of hidden states from the transformer.
                - attentions (Tuple): A tuple of attention weights from the transformer.

        Raises:
            None.
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)

        lm_loss, mc_loss = None, None
        if mc_labels is not None:
            mc_loss = F.cross_entropy(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        output = (lm_logits, mc_logits) + transformer_outputs[1:]
        if mc_loss is not None:
            output = (mc_loss,) + output
        if lm_loss is not None:
            output = (lm_loss,) + output
        return output

class GPTForSequenceClassification(GPTPreTrainedModel):
    """
    The Original GPT Model transformer with a sequence classification head on top (linear layer).
    GPTForSequenceClassification uses the last token in order to do the classification, as other causal
    models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the
    last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding
    token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since
    it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take
    the last value in each row of the batch).
    """
    def __init__(self, config):
        """
        Initializes a new instance of the GPTForSequenceClassification class.

        Args:
            self (object): The instance of the class.
            config (object):
                An object containing configuration settings for the model.

                - Type: dict
                - Purpose: Contains the necessary parameters for configuring the model.
                - Restrictions: Must include the following keys:

                    - num_labels: The number of labels for classification.
                    - pad_token_id: The token ID used for padding sequences.
                    - problem_type: The type of problem to solve (optional).

        Returns:
            None.

        Raises:
            None. This method does not raise any exceptions.
        """
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.transformer = GPTModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        self.pad_token_id = self.config.pad_token_id
        problem_type = config.problem_type
        if problem_type is None:
            self.loss = None
        else:
            if self.num_labels == 1:
                self.problem_type = "regression"
                self.loss = nn.MSELoss()
            elif self.num_labels > 1:
                self.problem_type = "single_label_classification"
                self.loss = nn.CrossEntropyLoss()
            else:
                self.problem_type = "multi_label_classification"
                self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in
                `[0, ...,config.num_labels - 1]`.
                If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, _ = input_ids.shape[:2]
        else:
            batch_size, _ = inputs_embeds.shape[:2]

        # Ensure the batch size is > 1 if there is no padding.
        if self.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # reduce sum not support int on Ascend.
                sequence_lengths = ops.ne(input_ids, self.pad_token_id) \
                        .astype(mindspore.float32).sum(-1) \
                        .astype(mindspore.int32) - 1
            else:
                sequence_lengths = -1

        pooled_logits = logits[ops.arange(batch_size), sequence_lengths]

        loss = None

        output = (pooled_logits,) + transformer_outputs[1:]

        if labels is not None:
            if self.num_labels == 1:
                loss = self.loss(pooled_logits.squeeze(), labels.squeeze())
            elif self.num_labels > 1:
                loss = self.loss(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = self.loss(pooled_logits, labels)

        if loss is not None:
            output = (loss,) + output
        return output


__all__ = ['GPTModel', 'GPTLMHeadModel',
           'GPTDoubleHeadsModel', 'GPTForSequenceClassification']
