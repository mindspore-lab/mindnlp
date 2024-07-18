# coding=utf-8
# Copyright 2021 The Eleuther AI and HuggingFace Inc. team. All rights reserved.
# Copyright 2023 Huawei Technologies Co., Ltd

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
# ============================================================================
"""
MindSpore BaiChuan Model
"""

import math
from typing import List, Optional, Tuple, Union
from queue import Queue
from threading import Thread


import numpy as np
import mindspore
from mindspore import Tensor, Parameter
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindspore import dtype as mstype
from mindnlp.utils import logging

from .configuration_baichuan import BaiChuanConfig
from ...generation.utils import GenerationConfig
from ...modeling_utils import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

logger = logging.get_logger(__name__)


def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    """ 
    This function builds the input tokens for a chat generation model.
    
    Args:
        model (object): The chat generation model object.
        tokenizer (object): The tokenizer object for encoding and decoding the messages.
        messages (List[dict]): A list of dictionaries representing the chat messages,
            where each dictionary contains 'role' (user or system) and 'content' (the message text).
        max_new_tokens (int, optional): The maximum number of new tokens that can be added to the input. Defaults to 0.
    
    Returns:
        None.
    
    Raises:
        AssertionError: If the 'role' in the messages is not correctly specified.
    """
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds

    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for round in rounds[::-1]:
        round_tokens = []
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return mindspore.tensor([input_tokens])


class TextIterStreamer:

    """
    The TextIterStreamer class represents a streamer for iterating over text data.
    It provides functionality for processing and streaming text data using a specified tokenizer.
    
    Parameters:
        tokenizer (object): The tokenizer to be used for processing the text data.
        skip_prompt (bool, optional): If set to True, prompts are skipped during iteration. Defaults to False.
        skip_special_tokens (bool, optional): If set to True, special tokens are skipped during iteration. Defaults to False.

    Attributes:
        tokenizer (object): The specified tokenizer for processing the text data.
        skip_prompt (bool): Indicates whether prompts are skipped during iteration.
        skip_special_tokens (bool): Indicates whether special tokens are skipped during iteration.
        tokens (list): A list to store the processed tokens.
        text_queue (Queue): A queue to store the processed text data.
        next_tokens_are_prompt (bool): Indicates whether the next tokens are prompts.

    Methods:
        put(self, value): Adds the processed value to the token list and the text queue.
        end(self): Signals the end of text data processing.
        __iter__(self): Returns the iterator object.
        __next__(self): Retrieves the next processed text data.

    Raises:
        - StopIteration: When the end of text data processing is reached.

    Note:
        - The put method processes the input value and adds it to the token list and text queue.
        If skip_prompt is set to True, prompts are skipped during processing.
        The end method signals the end of text data processing, and the __next__ method retrieves the next processed text data.

    Example:
        ```python
        >>> # Create a TextIterStreamer instance
        >>> streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        ...
        >>> # Add processed text data
        >>> streamer.put(processed_value)
        ...
        >>> # Signal the end of text data processing
        >>> streamer.end()
        ...
        >>> # Iterate over processed text data
        >>> for text_data in streamer:
        >>>     print(text_data)
        ```

    """
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        """
        Initializes a new instance of the TextIterStreamer class.

        Args:
            self (TextIterStreamer): The current instance of the TextIterStreamer class.
            tokenizer (object): The tokenizer object used for tokenization.
            skip_prompt (bool): A flag indicating whether to skip the prompt during tokenization.
            skip_special_tokens (bool): A flag indicating whether to skip special tokens during tokenization.

        Returns:
            None.

        Raises:
            None.

        This method initializes the TextIterStreamer class by setting the tokenizer, skip_prompt,
        and skip_special_tokens attributes.
        It also initializes the tokens and text_queue attributes.
        The tokenizer  object is used for tokenization.
        The skip_prompt flag is used to determine whether the prompt should be skipped during tokenization.
        The skip_special_tokens flag is used to determine whether special tokens  should be skipped during tokenization.
        The tokens attribute is an empty list that will store the tokens.
        The text_queue attribute is a queue used for storing the text.
        """
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Args:
            self: TextIterStreamer
                The instance of the TextIterStreamer class.
            value: array-like
                The input value to be added to the tokens. It should be an array-like object containing the tokens to be added.

        Returns:
            None.

        Raises:
            None
        """
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(
                self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        """
        Ends the text iteration process by putting a None value into the text queue.

        Args:
            self (TextIterStreamer): An instance of the TextIterStreamer class.

        Returns:
            None.

        Raises:
            None.

        Note:
            The method is used to signal the end of the text iteration process by putting a None
            value into the text queue. This allows other parts of the program to detect the end of
            the iteration and take appropriate actions.
        """
        self.text_queue.put(None)

    def __iter__(self):
        """
        __iter__ method in the TextIterStreamer class.

        Args:
            self: An instance of the TextIterStreamer class.
                This parameter represents the instance of the TextIterStreamer class on which the method is being called.
                It is required for accessing the attributes and methods within the class.

        Returns:
            None:
                This method does not return any value explicitly.
                Instead, it returns the instance of the TextIterStreamer class itself.

        Raises:
            This method does not raise any exceptions explicitly.
        """
        return self

    def __next__(self):
        """
        Docstring for the '__next__' method in the 'TextIterStreamer' class.

        Args:
            self: (TextIterStreamer) The instance of the TextIterStreamer class.
                It represents the current object on which the method is being called.

        Returns:
            None.

        Raises:
            StopIteration: If the value retrieved from the text_queue is None,
                this exception is raised to signal the end of iteration.
        """
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        return value

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: Union[tuple, list], dtype: mstype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full(
        (tgt_len, tgt_len),
        Tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min, dtype),
    )
    mask_cond = ops.arange(mask.shape[-1])
    mask = ops.masked_fill(mask, Tensor(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1)), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.concat(
            [ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1
        )
    return ops.broadcast_to(
        mask[None, None, :, :], (bsz, 1, tgt_len, tgt_len + past_key_values_length)
    )


def _expand_mask(mask: Tensor, dtype: mstype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = ops.broadcast_to(
        mask[:, None, None, :], (bsz, 1, tgt_len, src_len)
    ).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(mindspore.bool_),
        mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(dtype)).min),
    )

def _get_interleave(n):
    """
    This function returns an interleaved list of length n.

    Args:
        n (int): The length of the interleaved list to be generated. It must be a positive integer.

    Returns:
        list: Returns a list of length n containing interleaved values.

    Raises:
        ValueError: If the input parameter n is not a positive integer.
    """
    def _get_interleave_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_interleave_power_of_2(n)
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return _get_interleave_power_of_2(closest_power_of_2) + \
            _get_interleave(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

def _fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill(float("-inf")).astype(t.dtype)

def _gen_alibi_mask(n_head, max_pos):
    """used in inference only"""
    slopes = mindspore.Tensor(_get_interleave(n_head))
    alibi = slopes.unsqueeze(1).unsqueeze(1) * ops.arange(max_pos).unsqueeze(0).unsqueeze(0).broadcast_to(
        (n_head, -1, -1))
    alibi = alibi.view(n_head, 1, max_pos)
    alibi_mask = ops.triu(
        _fill_with_neg_inf(ops.zeros((max_pos, max_pos))), 1
    )
    alibi_mask = alibi_mask.unsqueeze(0) + alibi
    return alibi_mask

def _buffered_future_mask(tensor, maxpos, alibi, attn_heads):
    """used in training only"""
    _future_mask = ops.triu(
        _fill_with_neg_inf(ops.zeros([maxpos, maxpos])), 1
    )
    _future_mask = _future_mask.unsqueeze(0) + alibi
    _future_mask = _future_mask.to(tensor)
    return _future_mask[:tensor.shape[0] * attn_heads, :maxpos, :maxpos]


class RMSNorm(nn.Cell):
    """
    RMSNorm
    """
    def __init__(self, hidden_size, epsilon=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = Parameter(ops.ones(hidden_size), 'weight')
        self.variance_epsilon = epsilon

    def construct(self, hidden_states):
        """
        This method constructs RMSNorm by normalizing the hidden states.

        Args:
            self (RMSNorm): The instance of the RMSNorm class.
            hidden_states (Tensor):
                The input hidden states to be normalized. Should be a tensor of shape (batch_size, hidden_size).

        Returns:
            None.

        Raises:
            ValueError: If the hidden_states tensor is not of the correct shape.
            TypeError: If the data type of self.weight is not mindspore.float16 or mindspore.bfloat16.
        """
        variance = hidden_states.to(mindspore.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [mindspore.float16, mindspore.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class RotaryEmbedding(nn.Cell):
    """
    RotaryEmbedding
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        """
        __init__(self, dim, max_position_embeddings=2048, base=10000)

        Initialize the RotaryEmbedding object.

        Args:
            self: The instance of the class.
            dim (int): The dimensionality of the input data.
            max_position_embeddings (int, optional): The maximum sequence length for position embeddings. Defaults to 2048.
            base (int, optional): The base value used in the calculation. Defaults to 10000.

        Returns:
            None.

        Raises:
            ValueError: If the input parameters are not within the expected range or format.
            TypeError: If the input parameters are not of the expected type.
            RuntimeError: If there is an issue with the operations or calculations performed within the method.
        """
        super().__init__()
        self.inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2).float() / dim))

        self.max_seq_len_cached = max_position_embeddings
        t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = ops.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = ops.cat((freqs, freqs), axis=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def construct(self, x, seq_len=None):
        """
        Constructs the rotary embedding for a given sequence length.

        Args:
            self (RotaryEmbedding): An instance of the RotaryEmbedding class.
            x: The input tensor of shape (batch_size, ..., seq_len, ...).
            seq_len (int): The length of the input sequence. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If seq_len is greater than the maximum sequence length cached.

        This method constructs the rotary embedding for a given sequence length.
        It updates the cached cosine and sine values based on the input sequence length.

        If the input sequence length (seq_len) is greater than the maximum sequence length cached (self.max_seq_len_cached),
        the method computes the cosine and sine values for the updated sequence length.
        It then updates the cached cosine (self.cos_cached) and sine (self.sin_cached) values accordingly.

        The rotary embedding is computed using the inverse frequency tensor (self.inv_freq).
        The frequency tensor is generated by multiplying the sequence length with the inverse frequency tensor.
        The resulting tensor is concatenated with itself along the last axis to create the embedding tensor (emb).

        The cosine and sine values are computed from the embedding tensor using the cos and sin functions, respectively.
        The resulting cosine and sine tensors are then stored in the cached variables (self.cos_cached and self.sin_cached)
        to be reused in subsequent calls.

        Finally, the method returns the cosine and sine tensors sliced to the specified sequence length (seq_len),
        converted to the same data type as the input tensor (x.dtype).

        Note:
            The maximum sequence length cached (self.max_seq_len_cached) is updated only when the input sequence length
            (seq_len) is greater than it.
            This ensures that the cached values are reused for shorter sequences, improving efficiency.
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = ops.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = ops.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = ops.cat((freqs, freqs), axis=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return ops.cat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary positional embeddings to input queries (q) and keys (k).
    """
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLP(nn.Cell):
    """
    MLP
    """
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        """
        __init__ method for the MLP class.

        Args:
            self: Reference to the current instance of the class.
            hidden_size (int): The size of the hidden layer in the MLP.
                Specifies the number of neurons in the hidden layer.
            intermediate_size (int): The size of the intermediate layer in the MLP.
                Specifies the number of neurons in the intermediate layer.
            hidden_act (str): The activation function for the hidden layer.
                Specifies the activation function to be used in the hidden layer.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.gate_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(intermediate_size, hidden_size, has_bias=False)
        self.up_proj = nn.Dense(hidden_size, intermediate_size, has_bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def construct(self, x):
        """
        Method 'construct' in class 'MLP' constructs a multi-layer perceptron.

        Args:
            self (object): The instance of the class.
            x (tensor): The input tensor to be processed by the MLP.

        Returns:
            None: The method modifies the internal state of the MLP object.

        Raises:
            None.
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config: BaiChuanConfig):
        """
        Initializes an instance of the Attention class.

        Args:
            self: The instance of the class.
            config (BaiChuanConfig):
                An object of type BaiChuanConfig containing configuration parameters for the attention mechanism.

                - The 'config' parameter is required.
                - It specifies the configuration settings for the attention mechanism.
                - The 'config' parameter must be of type BaiChuanConfig.

        Returns:
            None

        Raises:
            ValueError:
                Raised if 'hidden_size' is not divisible by 'num_heads'.

                - This exception is raised when the size of the hidden state is not evenly divisible
                by the number of attention heads.
                - The 'hidden_size' and 'num_heads' parameters must satisfy this condition.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.W_pack = nn.Dense(self.hidden_size, 3 * self.hidden_size, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        """
        This method reshapes the input tensor to match the specified dimensions for
        batch size, sequence length, number of heads, and head dimension.

        Args:
            tensor (Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence.
            bsz (int): The batch size.

        Returns:
            None: This method returns None as the reshaped tensor is directly modified in place.

        Raises:
            ValueError: If the input tensor does not have the required dimensions.
            TypeError: If the input tensor is not of type Tensor, or if seq_len and bsz are not integers.
            RuntimeError: If the method encounters a runtime error while reshaping the tensor.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        """
        Method 'construct' in the class 'Attention' processes hidden states using self-attention mechanism.

        Args:
            self: The instance of the Attention class.
            hidden_states (Tensor): Input hidden states of shape (batch size, sequence length, hidden size).
            attention_mask (Optional[Tensor]): Masking tensor to control which tokens can attend to which other tokens.
            position_ids (Optional[Tensor]): Tensor indicating the position of each token in the sequence.
            past_key_value (Optional[Tuple[Tensor]]): Tuple containing past key and value states for caching.
            output_attentions (bool): Flag to determine whether to output attention weights.
            use_cache (bool): Flag to indicate if caching of key and value states should be used.

        Returns:
            Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
                The output of the attention mechanism, attention weights (if output_attentions is True),
                and updated key and value states for caching.

        Raises:
            ValueError: If the shape of attention weights is incorrect.
            ValueError: If the shape of attention mask is incorrect.
        """
        bsz, q_len, _ = hidden_states.shape

        proj = self.W_pack(hidden_states)
        m = nn.Unflatten(-1, (3, self.hidden_size))
        proj = m(proj).unsqueeze(0).swapaxes(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)  # batch_size x source_len x hidden_size
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)  # batch_size x target_len x head_size
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)  # batch_size x source_len x hidden_size

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)
        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = ops.maximum(attn_weights,
                                       Tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = ops.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class BaiChuanAttention(nn.Cell):

    """
    BaiChuanAttention class represents an attention mechanism component used in neural network models.
    It is designed to calculate attention weights and apply them to the input hidden states to generate the
    final output.
    This class inherits from nn.Cell.

    Attributes:
        config (BaiChuanConfig): An instance of BaiChuanConfig containing configuration parameters for the attention mechanism.
        hidden_size (int): The size of the hidden state vectors.
        num_heads (int): The number of attention heads used in the attention mechanism.
        head_dim (int): The dimension of each attention head.
        max_position_embeddings (int): The maximum length of the input sequence.
        W_pack (nn.Dense): A dense layer used for linear transformation.
        o_proj (nn.Dense): A dense layer used for projecting the attention output.

    Methods:
        __init__: Initializes the BaiChuanAttention instance with the provided configuration.
        _shape: Reshapes the input tensor into the desired shape for further processing.
        construct: Constructs the attention mechanism by calculating attention weights
            and output based on the input hidden states and additional parameters.

    Raises:
        ValueError: If the hidden size is not divisible by the number of attention heads.

    Returns:
        Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
            A tuple containing the attention output tensor, attention weights (if requested), and past key-value states (if
            caching is enabled).
    """
    def __init__(self, config: BaiChuanConfig):
        """
        Initializes an instance of the BaiChuanAttention class.

        Args:
            self: The instance of the class.
            config (BaiChuanConfig): The configuration object for BaiChuanAttention.
                This object contains various parameters and settings for the attention mechanism.
                It is expected to have the following attributes:

                - hidden_size (int): The size of the hidden state.
                - num_attention_heads (int): The number of attention heads.
                - model_max_length (int): The maximum length of the model.
                This is used for positional embeddings.

        Returns:
            None.

        Raises:
            ValueError: If the hidden_size is not divisible by the num_attention_heads.

        Note:
            This method initializes the BaiChuanAttention object by setting its configuration attributes.
            It also initializes the W_pack and o_proj layers using the hidden_size and num_attention_heads values.
            The W_pack layer is a fully connected layer with the input size of hidden_size and output size of 3 times hidden_size.
            The o_proj layer is a fully connected layer with the input size of num_attention_heads times head_dim and output size of hidden_size.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.model_max_length

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )
        self.W_pack = nn.Dense(self.hidden_size, 3 * self.hidden_size, has_bias=False)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)

    def _shape(self, tensor: mindspore.Tensor, seq_len: int, bsz: int):
        """
        Reshapes the input tensor for the BaiChuanAttention module.

        Args:
            self (BaiChuanAttention): The instance of the BaiChuanAttention class.
            tensor (mindspore.Tensor): The input tensor to be reshaped.
            seq_len (int): The length of the sequence in the tensor.
            bsz (int): The batch size of the tensor.

        Returns:
            None.

        Raises:
            None.

        Description:
            This method reshapes the input tensor to match the required shape for BaiChuanAttention module.
            It rearranges the dimensions of the tensor according to the number of heads and the head dimension.
            The dimensions are rearranged in the following order: (bsz, seq_len, num_heads, head_dim).

        Example:
            ```python
            >>> # Create an instance of BaiChuanAttention
            >>> attention = BaiChuanAttention()
            ...
            >>> # Create a tensor
            >>> tensor = mindspore.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            ...
            >>> # Reshape the tensor using the _shape method
            >>> attention._shape(tensor, 5, 2)
            ```
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
        '''
        Constructs the attention mechanism for the BaiChuanAttention class.

        Args:
            self (BaiChuanAttention): An instance of the BaiChuanAttention class.
            hidden_states (mindspore.Tensor):
                The hidden states tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[mindspore.Tensor]):
                An optional tensor of shape (batch_size, sequence_length) or (batch_size, sequence_length, sequence_length).
                It is used as a mask to prevent attention to certain positions. Defaults to None.
            past_key_value (Optional[Tuple[mindspore.Tensor]]):
                An optional tuple containing the past key and value states tensors of shape (batch_size, past_sequence_length, hidden_size).
                If provided, the attention mechanism will incorporate the past key and value states. Defaults to None.
            output_attentions (bool): A flag indicating whether to return the attention weights. Defaults to False.
            use_cache (bool): A flag indicating whether to cache the key and value states for future use. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[mindspore.Tensor], Optional[Tuple[mindspore.Tensor]]]:
                A tuple containing

                - the attention output tensor of shape (batch_size, sequence_length, hidden_size),
                - an optional tensor representing the attention weights of shape (batch_size, num_heads, sequence_length, sequence_length),
                - and an optional tuple containing the key and value states tensors of shape (batch_size, sequence_length, hidden_size).

        Raises:
            None.
        '''
        bsz, q_len, _ = hidden_states.shape

        proj = self.W_pack(hidden_states)
        proj = proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).swapaxes(0, -2).squeeze(-2)
        query_states = proj[0].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = proj[1].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        value_states = proj[2].view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = ops.cat([past_key_value[0], key_states], axis=2)
            value_states = ops.cat([past_key_value[1], value_states], axis=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if q_len == 1: # inference with cache
                if len(attention_mask.shape) == 4:
                    attention_mask = attention_mask[:, :, -1:, :]
                else:
                    attention_mask = attention_mask[:, -1:, :]
            attn_weights = attn_weights + attention_mask.astype(attn_weights.dtype)
            attn_weights = ops.maximum(attn_weights, mindspore.tensor(np.finfo(mindspore.dtype_to_nptype(attn_weights.dtype)).min))

        attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_output = ops.matmul(attn_weights, value_states)

        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DecoderLayer(nn.Cell):
    """
    DecoderLayer
    """
    def __init__(self, config: BaiChuanConfig):
        """
        Initializes an instance of the DecoderLayer class.

        Args:
            self: The instance of the DecoderLayer class.
            config (BaiChuanConfig):
                The configuration object containing various settings for the DecoderLayer.

                - hidden_size (int): The size of the hidden state.
                - intermediate_size (int): The size of the intermediate layer in the MLP.
                - hidden_act (str): The activation function to be used in the hidden layer of the MLP.
                - rms_norm_eps (float): The epsilon value for RMSNorm.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: Tensor,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_value: Optional[Tuple[Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Args:
            hidden_states (`Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BaiChuanLayer(nn.Cell):

    '''
    The BaiChuanLayer class represents a layer used for implementing a specific type of neural network cell.
    This class inherits from the nn.Cell class.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        self_attn (BaiChuanAttention): An instance of the BaiChuanAttention class for self-attention mechanism.
        mlp (MLP): An instance of the MLP class for multi-layer perceptron operations.
        input_layernorm (RMSNorm): An instance of the RMSNorm class for input layer normalization.
        post_attention_layernorm (RMSNorm): An instance of the RMSNorm class for post-attention layer normalization.

    Methods:
        __init__: Initializes the BaiChuanLayer class.
        construct:
            Constructs the BaiChuanLayer with the given parameters and returns the hidden states with
            optional present key value computed during attention.

    Raises:
        None

    Returns:
        Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
            A tuple containing the hidden states and optional present key value.

    Note:
        This class is designed to be used as a part of a neural network model for specific tasks.

    '''
    def __init__(self, config: BaiChuanConfig):
        """
        Initializes a new instance of the BaiChuanLayer class.

        Args:
            self: The instance of the BaiChuanLayer class.
            config (BaiChuanConfig): An instance of BaiChuanConfig containing the configuration parameters for the layer.
                It specifies the hidden size, intermediate size, hidden activation function, and epsilon for RMS normalization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = BaiChuanAttention(config=config)
        self.mlp = MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def construct(
            self,
            hidden_states: mindspore.Tensor,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_value: Optional[Tuple[mindspore.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
        """
        Constructs the BaiChuanLayer.

        This method applies a series of transformations to the input hidden states to generate the output tensor.

        Args:
            self (BaiChuanLayer): The instance of the BaiChuanLayer class.
            hidden_states (mindspore.Tensor): The input hidden states tensor.
            attention_mask (Optional[mindspore.Tensor]): The optional attention mask tensor. Defaults to None.
            past_key_value (Optional[Tuple[mindspore.Tensor]]): The optional tuple of past key and value tensors. Defaults to None.
            output_attentions (Optional[bool]): Whether to output the attentions. Defaults to False.
            use_cache (Optional[bool]): Whether to use cache. Defaults to False.

        Returns:
            Tuple[mindspore.Tensor, Optional[Tuple[mindspore.Tensor, mindspore.Tensor]]]:
                The output tensor and the optional tuple of present key and value tensors.

        Raises:
            None.

        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BaiChuanPreTrainedModel(PreTrainedModel):
    """
    BaiChuanPreTrainedModel
    """
    config_class = BaiChuanConfig
    base_model_prefix = "model"
    _no_split_modules = ["DecoderLayer", "BaiChuanLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, cell):
        """
        Initializes the weights for the given cell.

        Args:
            self (BaiChuanPreTrainedModel): The instance of the BaiChuanPreTrainedModel class.
            cell: The cell for which the weights need to be initialized.

        Returns:
            None.

        Raises:
            None.
        """
        std = self.config.initializer_range
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Normal(
                sigma=std, mean=0.0), cell.weight.shape, cell.weight.dtype))
            if cell.bias is not None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, std, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0

            cell.weight.set_data(Tensor(weight, cell.weight.dtype))


class BaiChuan7bModel(BaiChuanPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DecoderLayer`]
    Args:
        config: BaiChuanConfig
    """
    def __init__(self, config: BaiChuanConfig):
        """
        Initializes a new instance of the BaiChuan7bModel class.

        Args:
            self: The instance of the BaiChuan7bModel class.
            config (BaiChuanConfig):
                An instance of BaiChuanConfig containing configuration parameters.

                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be an instance of BaiChuanConfig.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input embeddings for the BaiChuan7bModel.

        Args:
            self: The instance of BaiChuan7bModel.

        Returns:
            None.

        Raises:
            None.

        This method retrieves the input embeddings for the BaiChuan7bModel.
        The input embeddings are obtained by calling the 'embed_tokens' method of the instance.
        """
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Sets the input embeddings for the BaiChuan7bModel.

        Args:
            self (BaiChuan7bModel): The instance of the BaiChuan7bModel class.
            new_embeddings (Any): The new embeddings to be set. This can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.embed_tokens = new_embeddings

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        """
        This method prepares the decoder attention mask based on the provided parameters.

        Args:
            self (BaiChuan7bModel): The instance of the BaiChuan7bModel class.
            attention_mask (torch.Tensor): The attention mask tensor to be applied during decoding.
                If None, no attention mask will be applied.
            input_shape (tuple): The shape of the input tensor (batch_size, sequence_length, hidden_size).
            inputs_embeds (torch.Tensor): The embedded input tensor of shape (batch_size, sequence_length, hidden_size).
            past_key_values_length (int): The length of past key values to consider for the attention mask.

        Returns:
            None: This method returns the combined attention mask or None if no attention mask is applied.

        Raises:
            ValueError: If the input_shape[-1] is less than or equal to 1.
            TypeError: If the input data types are incompatible for mask operations.
        """
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        This method constructs the BaiChuan7bModel by processing the input data and generating model outputs.

        Args:
            self (object): The instance of the class BaiChuan7bModel.
            input_ids (Tensor): The input tensor containing token indices representing the input sequence. Default is None.
            attention_mask (Optional[Tensor]): Optional tensor specifying the attention mask for the input sequence. Default is None.
            position_ids (Optional[Tensor]): Optional tensor specifying the position indices for the input sequence. Default is None.
            past_key_values (Optional[List[Tensor]]): Optional list of tensors containing past key values for the model. Default is None.
            inputs_embeds (Optional[Tensor]): Optional tensor containing the embeddings of the input tokens. Default is None.
            use_cache (Optional[bool]): Optional boolean flag indicating whether to use cache during model computation. Default is None.
            output_attentions (Optional[bool]): Optional boolean flag indicating whether to output attentions. Default is None.
            output_hidden_states (Optional[bool]): Optional boolean flag indicating whether to output hidden states. Default is None.
            return_dict (Optional[bool]): Optional boolean flag indicating whether to return the output as a dictionary. Default is None.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]: Returns a tuple or BaseModelOutputWithPast object containing the model outputs.

        Raises:
            ValueError:
                Raised if both input_ids and inputs_embeds are specified simultaneously,
                if neither decoder_input_ids nor decoder_inputs_embeds are specified,
                or if an invalid configuration is encountered during model construction.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            position_ids = ops.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=mindspore.int64
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = ops.ones(
                (batch_size, seq_length_with_past), dtype=mindspore.bool_
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # TODO: how checkpoint
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BaiChuan13bModel(BaiChuanPreTrainedModel):

    """
    This class represents a BaiChuan13b model for natural language processing tasks. It is a subclass of the BaiChuanPreTrainedModel class.
    The BaiChuan13bModel class contains methods for initializing the model, getting and setting input embeddings,
    generating an alibi mask, and constructing the model.

    Attributes:
        padding_idx (int): The index used for padding tokens in the embedding layer.
        vocab_size (int): The size of the vocabulary.
        n_head (int): The number of attention heads.
        embed_tokens (nn.Embedding): The embedding layer for input tokens.
        layers (nn.CellList): A list of BaiChuanLayer instances representing the layers of the model.
        norm (RMSNorm): The normalization layer applied after the model layers.
        max_cache_pos (int): The maximum position of past key values for caching.
        first_run (bool): A flag indicating if it is the first run of the model.
        alibi_mask (Optional[mindspore.Tensor]): A tensor representing the alibi mask.

    Methods:
        __init__(self, config: BaiChuanConfig): Initializes the BaiChuan13bModel instance with a configuration.
        get_input_embeddings(self): Returns the input embeddings of the model.
        set_input_embeddings(self, value): Sets the input embeddings of the model.
        get_alibi_mask(self, tensor, seq_length_with_past): Generates an alibi mask based on the tensor and sequence length.
        construct(self, input_ids, attention_mask, past_key_values, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict):
            Constructs the model with the given inputs and returns the model output.

    Note:
        - The BaiChuan13bModel class is designed to be used for natural language processing tasks, such as text classification or language generation.
        - The model architecture follows the BaiChuan13b configuration, which includes embedding layers, multiple layers of BaiChuanLayer, and normalization layers.
        - The alibi mask is used for attention calculations and is generated based on the input tensor and sequence length.
        - The construct method is the main entry point for using the model, which takes various inputs and returns the model output.
    """
    def __init__(self, config: BaiChuanConfig):
        """
        __init__

        This method initializes an instance of the BaiChuan13bModel class.

        Args:
            self: The instance of the BaiChuan13bModel class.
            config (BaiChuanConfig):
                An object of type BaiChuanConfig containing configuration parameters for the model.
                It specifies the configuration parameters such as pad_token_id, vocab_size,
                num_attention_heads, hidden_size, num_hidden_layers, rms_norm_eps, and model_max_length.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.n_head = config.num_attention_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList([BaiChuanLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.post_init()
        self.max_cache_pos = config.model_max_length
        self.first_run = True
        self.alibi_mask = None

    def get_input_embeddings(self):
        """
        This method returns the input embeddings for the BaiChuan13bModel.

        Args:
            self: The instance of the BaiChuan13bModel class.

        Returns:
            None: This method returns the input embeddings for the BaiChuan13bModel as an instance of 'embed_tokens'.

        Raises:
            None
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Method to set the input embeddings for the BaiChuan13bModel.

        Args:
            self (BaiChuan13bModel): The instance of the BaiChuan13bModel class.
            value:
                The input embeddings to be set for the model.

                - Type: Any
                - Purpose: Represents the new input embeddings to be assigned to the model.
                - Restrictions: None

        Returns:
            None.

        Raises:
            None
        """
        self.embed_tokens = value

    def get_alibi_mask(self, tensor, seq_length_with_past):
        """
        This method is a member of the 'BaiChuan13bModel' class and is used to obtain an alibi mask
        based on the input tensor and sequence length with past information.

        Args:
            self (object): The instance of the class.
            tensor (Tensor): The input tensor used to derive the alibi mask.
            seq_length_with_past (int): The length of the sequence with past information.

        Returns:
            None.

        Raises:
            ValueError: If the 'seq_length_with_past' parameter is not an integer.
            RuntimeError: If the method encounters issues during execution.
        """
        if self.training:
            slopes = mindspore.Tensor(_get_interleave(self.n_head))
            alibi = slopes.unsqueeze(1).unsqueeze(1) * ops.arange(seq_length_with_past).unsqueeze(0).unsqueeze(0).broadcast_to(
                (self.n_head, -1, -1))
            alibi = alibi.view(self.n_head, 1, seq_length_with_past)
            mask = _buffered_future_mask(tensor, seq_length_with_past, alibi, self.n_head)
        else:
            if self.first_run:
                self.first_run = False
                self.future_mask = _gen_alibi_mask(self.n_head, self.max_cache_pos)
            if seq_length_with_past > self.max_cache_pos:
                self.max_cache_pos = seq_length_with_past
                self.future_mask = _gen_alibi_mask(self.n_head, self.max_cache_pos)
            mask = self.future_mask[:self.n_head, :seq_length_with_past, :seq_length_with_past]
        return mask

    def construct(
            self,
            input_ids: mindspore.Tensor = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            past_key_values: Optional[List[mindspore.Tensor]] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Constructs the BaiChuan13bModel.

        Args:
            self: The object instance.
            input_ids (mindspore.Tensor, optional): The input tensor of shape [batch_size, sequence_length].
            attention_mask (mindspore.Tensor, optional): The attention mask tensor of shape [batch_size, sequence_length].
            past_key_values (List[mindspore.Tensor], optional): The list of past key value tensors.
            inputs_embeds (mindspore.Tensor, optional): The input embeddings tensor of shape [batch_size, sequence_length, hidden_size].
            use_cache (bool, optional): Whether to use cache for decoding.
            output_attentions (bool, optional): Whether to output attention weights.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary instead of a tuple.

        Returns:
            Union[Tuple, BaseModelOutputWithPast]:
                The output tuple or BaseModelOutputWithPast object containing the last hidden state,
                past key values, hidden states, and attentions.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided simultaneously.
            ValueError: If neither input_ids nor inputs_embeds are provided.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds simultaneously")
        if input_ids is not None:
            _, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You need to provide input_ids or inputs_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        seq_length_with_past = seq_length

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.training:
            if self.alibi_mask is None or self.alibi_mask.shape[-1] != seq_length_with_past:
                self.alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)
            alibi_mask = self.alibi_mask
        else:
            alibi_mask = self.get_alibi_mask(inputs_embeds, seq_length_with_past)

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                expanded_mask = attention_mask.to(alibi_mask.dtype)
                expanded_mask = ops.tril(ops.gt(expanded_mask[:, :, None] * expanded_mask[:, None, :], 0)
                                ) * ops.eq(expanded_mask[:, :, None] - expanded_mask[:, None, :], 0)
            else:
                expanded_mask = attention_mask
            bsz = inputs_embeds.size(0)
            src_len, tgt_len = alibi_mask.shape[-2:]
            expanded_mask = expanded_mask.unsqueeze(1).broadcast_to((bsz, 1, src_len, tgt_len)).to(alibi_mask.dtype)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask = inverted_mask.masked_fill(inverted_mask.to(mindspore.bool_), np.finfo(mindspore.dtype_to_nptype(alibi_mask.dtype)).min)
            attention_mask = inverted_mask + alibi_mask.unsqueeze(0)
        else:
            attention_mask = alibi_mask

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class BaiChuanForCausalLM(BaiChuanPreTrainedModel):
    """
    BaiChuanForCausalLM
    """
    def __init__(self, config, size=None):
        """
        Initializes a new instance of BaiChuanForCausalLM.

        Args:
            self: The instance of the class.
            config: The configuration for the model.
            size (str, optional): The size of the model. Defaults to None. Must be either '7b' or '13b'.

        Returns:
            None.

        Raises:
            ValueError:
                If the size parameter is not '7b' or '13b', a ValueError is raised with the message 'BaiChuan model
                only supports 7b and 13b, please check your config.'
        """
        super().__init__(config)
        if size == '7b':
            self.model = BaiChuan7bModel(config)
        elif size == '13b':
            self.model = BaiChuan13bModel(config)
        else:
            self.model = BaiChuan7bModel(config)
            raise ValueError('BaiChuan model only support 7b and 13b, please check your config.')

        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Method to retrieve the input embeddings from the model for the BaiChuanForCausalLM class.

        Args:
            self: The instance of the BaiChuanForCausalLM class. It is used to access the model's embed_tokens.

        Returns:
            None: This method returns the input embeddings from the model to be used in the BaiChuanForCausalLM class.

        Raises:
            None.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        """
        Set the input embeddings for the BaiChuanForCausalLM model.

        Args:
            self (BaiChuanForCausalLM): The instance of the BaiChuanForCausalLM class.
            new_embeddings (torch.Tensor): The new input embeddings to be set for the model. Should be of shape (vocab_size, embedding_dim).

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of type torch.Tensor.
            ValueError: If the shape of new_embeddings does not match the expected shape (vocab_size, embedding_dim).
        """
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        """
        This method retrieves the output embeddings from the BaiChuanForCausalLM model.

        Args:
            self: An instance of the BaiChuanForCausalLM class.

        Returns:
            lm_head: The method returns the lm_head attribute which contains the output embeddings.

        Raises:
            None.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings for BaiChuanForCausalLM model.

        Args:
            self (BaiChuanForCausalLM): The instance of BaiChuanForCausalLM class.
            new_embeddings (Any): The new embeddings to be set as the output embeddings for the model.
                This can be of any type.

        Returns:
            None.

        Raises:
            None.
        """
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        """
        set_decoder
        """
        self.model = decoder

    def get_decoder(self):
        """
        get_decoder
        """
        return self.model

    def construct(
            self,
            input_ids: Tensor = None,
            attention_mask: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            past_key_values: Optional[List[Tensor]] = None,
            inputs_embeds: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Constructs the Causal Language Model for the BaiChuan model.

        Args:
            self (BaiChuanForCausalLM): The instance of the BaiChuanForCausalLM class.
            input_ids (Tensor, optional): The input tensor containing the token IDs. Default: None.
            attention_mask (Optional[Tensor], optional): The attention mask tensor. Default: None.
            position_ids (Optional[Tensor], optional): The position IDs tensor. Default: None.
            past_key_values (Optional[List[Tensor]], optional): The list of past key values tensor. Default: None.
            inputs_embeds (Optional[Tensor], optional): The input embeddings tensor. Default: None.
            labels (Optional[Tensor], optional): The tensor containing the labels. Default: None.
            use_cache (Optional[bool], optional): Whether to use cache. Default: None.
            output_attentions (Optional[bool], optional): Whether to output attentions. Default: None.
            output_hidden_states (Optional[bool], optional): Whether to output hidden states. Default: None.
            return_dict (Optional[bool], optional): Whether to return a dictionary. Default: None.

        Returns:
            Union[Tuple, CausalLMOutputWithPast]:
                The model outputs.

                - If `return_dict` is False, returns a tuple containing the logits and the various model outputs.
                - If `return_dict` is True, returns an instance of `CausalLMOutputWithPast` containing the loss,
                logits, past key values, hidden states, and attentions.

        Raises:
            ValueError: If the BaiChuan model is not of type BaiChuan7bModel or BaiChuan13bModel.

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if isinstance(self.model, BaiChuan7bModel):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif isinstance(self.model, BaiChuan13bModel):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            raise ValueError('BaiChuan model only support 7b and 13b, please check your config.')

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            loss = ops.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        This method prepares inputs for generation in the BaiChuanForCausalLM class.

        Args:
            self (object): The instance of the class.
            input_ids (torch.Tensor): The input token IDs. Shape (batch_size, sequence_length).
            past_key_values (tuple, optional):
                Tuple of tensors containing cached key and value projection states of the model. Default is None.
            attention_mask (torch.Tensor, optional):
                Mask to avoid performing attention on padding token indices. Shape (batch_size, sequence_length).
            inputs_embeds (torch.Tensor, optional):
                The embedded representation of the input tokens. Shape (batch_size, sequence_length, hidden_size).

        Returns:
            dict:
                A dictionary containing model inputs for generation, including 'input_ids', 'position_ids',
                'past_key_values', 'use_cache', and 'attention_mask'.

        Raises:
            ValueError: If attention_mask and position_ids are both provided and have mismatched shapes.
            ValueError: If inputs_embeds and past_key_values are both provided.
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorders the cache for the given beam indices.

        Args:
            past_key_values (tuple): A tuple containing the past key and value tensors for each layer.
                Each element in the tuple is a tuple of tensors representing the past states for the corresponding layer.
            beam_idx (tensor): A 1-D tensor containing the indices of the beams for reordering the cache.

        Returns:
            None: This method modifies the 'past_key_values' in place to reorder the cache based on the 'beam_idx'.

        Raises:
            ValueError: If the length of 'past_key_values' does not match the number of layers in the model.
            IndexError: If the 'beam_idx' contains indices that are out of range for the dimensions of
                the tensors in 'past_key_values'.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def chat(self, tokenizer, messages: List[dict], stream=False,
             generation_config: Optional[GenerationConfig]=None):
        """
        Method:
            chat

        Description:
            This method allows for conducting a chat conversation using the BaiChuanForCausalLM model.
            It takes in the necessary input parameters and returns the response generated by the model.

        Args:
            self: The instance of the BaiChuanForCausalLM class.
            tokenizer: An object of the tokenizer class used for tokenizing the input messages.
            messages: A list of dictionaries representing the chat messages.
                Each dictionary contains the following keys:

                - 'role': The role of the message sender (e.g., 'system', 'user', 'assistant').
                - 'content': The content of the message.
            stream: A boolean value indicating whether the chat conversation should be streamed or not. Default is False.
            generation_config: An optional object of the GenerationConfig class that specifies the generation configurations.
                If not provided, the instance's generation_config will be used.
        
        Returns:
            None
        
        Raises:
            None
        """
        generation_config = generation_config or self.generation_config
        input_ids = build_chat_input(self, tokenizer, messages, generation_config.max_new_tokens)
        if stream:
            streamer = TextIterStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=self.generate, kwargs={
                                                    "inputs": input_ids,
                                                    "streamer": streamer,
                                                    "generation_config": generation_config
                                                }
            ).start()
            return streamer

        outputs = self.generate(input_ids, generation_config=generation_config)
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return response


__all__ = [
    "BaiChuanPreTrainedModel",
    "BaiChuan7bModel",
    "BaiChuan13bModel",
    "BaiChuanForCausalLM"
]
