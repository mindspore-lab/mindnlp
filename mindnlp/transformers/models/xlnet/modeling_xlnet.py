# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
 PyTorch XLNet model.
"""
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from mindspore.common.initializer import initializer, Normal


import mindspore

from mindspore import Tensor, Parameter
from mindspore.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import mindnlp.core.nn.functional
from mindnlp. transformers. ms_utils import apply_chunking_to_forward
from mindnlp. transformers. activations import ACT2FN
from mindnlp. transformers. modeling_utils import PoolerAnswerClass, PoolerEndLogits, PoolerStartLogits, PreTrainedModel, SequenceSummary
from mindnlp.utils import (ModelOutput, logging)
from mindnlp.core import nn,ops
from .configuration_xlnet import XLNetConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "xlnet/xlnet-base-cased"
_CONFIG_FOR_DOC = "XLNetConfig"

XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlnet/xlnet-base-cased",
    "xlnet/xlnet-large-cased",
    # See all XLNet models at https://huggingface.co/models?filter=xlnet
]


class XLNetRelativeAttention(nn.Module):

    """This class represents the relative attention mechanism used in XLNet model for sequence processing tasks.

    The XLNetRelativeAttention class implements the core operations for performing relative positional attention
    in the XLNet model. It includes methods for initializing the attention mechanism, pruning attention heads,
    shifting for relative attention score calculation, and processing post-attention outputs.

    Attributes:
        n_head (int): Number of attention heads.
        d_head (int): Dimensionality of each attention head.
        d_model (int): Dimensionality of the model.
        scale (float): Scaling factor for attention scores.
        q (Parameter): Query matrix for attention computation.
        k (Parameter): Key matrix for attention computation.
        v (Parameter): Value matrix for attention computation.
        o (Parameter): Output matrix for attention computation.
        r (Parameter): Relative position matrix.
        r_r_bias (Parameter): Relative position bias for rows.
        r_s_bias (Parameter): Relative position bias for segments.
        r_w_bias (Parameter): Relative position bias for columns.
        seg_embed (Parameter): Segment embedding matrix.
        layer_norm (LayerNorm): Layer normalization for model outputs.
        dropout (Dropout): Dropout layer for regularization.

    Methods:
        prune_heads: Method to prune specific attention heads (NotImplementedError).
        rel_shift: Static method to perform relative shift for attention score calculation.
        rel_shift_bnij: Static method to perform relative shift for attention score calculation with
            different axis.
        rel_attn_core: Method for core relative positional attention operations.
        post_attention: Method for post-attention processing.
        forward: Method for forwarding the attention mechanism with optional outputs.

    Note:
        This class inherits from nn.Module, which is a base class for neural network cells in the MindSpore framework.
    """
    def __init__(self, config):
        '''
        Initializes the XLNetRelativeAttention class.

        Args:
            self (XLNetRelativeAttention): The instance of the XLNetRelativeAttention class.
            config: An object containing configuration parameters for the XLNetRelativeAttention model.
                It should have the following attributes:

                - d_model (int): The hidden size of the model.
                - n_head (int): The number of attention heads.
                - d_head (int): The size of each attention head.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - dropout (float): The dropout rate.

        Returns:
            None.

        Raises:
            ValueError: If the hidden size (config.d_model) is not a multiple of the number of attention heads
                (config.n_head).
        '''
        super().__init__()

        if config.d_model % config.n_head != 0:
            raise ValueError(
                f"The hidden size ({config.d_model}) is not a multiple of the number of attention "
                f"heads ({config.n_head}"
            )

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / (config.d_head ** 0.5)

        self.q = Parameter(ops.zeros((config.d_model, self.n_head, self.d_head), dtype=mindspore.float32))
        self.k = Parameter(ops.zeros((config.d_model, self.n_head, self.d_head), dtype=mindspore.float32))
        self.v = Parameter(ops.zeros((config.d_model, self.n_head, self.d_head), dtype=mindspore.float32))
        self.o = Parameter(ops.zeros((config.d_model, self.n_head, self.d_head), dtype=mindspore.float32))
        self.r = Parameter(ops.zeros((config.d_model, self.n_head, self.d_head), dtype=mindspore.float32))

        self.r_r_bias = Parameter(ops.zeros((self.n_head, self.d_head), dtype=mindspore.float32))
        self.r_s_bias = Parameter(ops.zeros((self.n_head, self.d_head), dtype=mindspore.float32))
        self.r_w_bias = Parameter(ops.zeros((self.n_head, self.d_head), dtype=mindspore.float32))
        self.seg_embed = Parameter(ops.zeros((2, self.n_head, self.d_head), dtype=mindspore.float32))

        self.layer_norm = nn.LayerNorm([config.d_model], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout)

    def prune_heads(self, heads):
        """
        This method prunes the given heads in the XLNetRelativeAttention class.

        Args:
            self (XLNetRelativeAttention): The instance of the XLNetRelativeAttention class.
            heads (int): The number of heads to be pruned from the attention mechanism. It should be a positive integer.

        Returns:
            None.

        Raises:
            NotImplementedError: This exception is raised if the method is called directly without being implemented
                in a subclass.
        """
        raise NotImplementedError

    @staticmethod
    def rel_shift(x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = x.shape

        x = x.reshape(x_size[1], x_size[0], x_size[2], x_size[3])
        x = x[1:, ...]
        x = x.reshape(x_size[0], x_size[1] - 1, x_size[2], x_size[3])
        # x = x[:, 0:klen, :, :]
        x = ops.index_select(x, 1, ops.arange(klen, dtype=mindspore.int64))

        return x

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        """
        Applies a relative shift to the input tensor along the batch and head dimensions in XLNetRelativeAttention class.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, head_size, hidden_size).
            klen (int, optional): Length of the relative attention. Defaults to -1.

        Returns:
            None

        Raises:
            TypeError: If x is not a Tensor.
            ValueError: If x has an invalid shape or klen is not within the valid range.

        Note:
            - This method applies a relative shift operation to the input tensor `x`, rearranging the elements along
            the batch and head dimensions.
            - The input tensor `x` should have shape (batch_size, seq_length, head_size, hidden_size).
            - The relative shift is performed by reshaping the tensor, selecting specific indices, and reshaping
            it back to the original shape.
            - The relative shift helps in capturing the relative positions of tokens in the attention mechanism.
            - If `klen` is not provided, the default value of -1 is used which indicates that the relative attention
            length is not restricted.

        Example:
            ```python
            >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
            >>> klen = 2
            >>> rel_shift_bnij(x, klen)
            ```
        """
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        x = ops.index_select(x, 3, ops.arange(klen, dtype=mindspore.int64))
        return x

    def rel_attn_core(
            self,
            q_head,
            k_head_h,
            v_head_h,
            k_head_r,
            seg_mat=None,
            attn_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        """Core relative positional attention operations."""

        ac = ops.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)


        bd = ops.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])


        if seg_mat is None:
            ef = 0
        else:
            ef = ops.einsum("ibnd,snd->ibns", q_head + self.r_s_bias, self.seg_embed)
            ef = ops.einsum("ijbs,ibns->bnij", seg_mat, ef)

        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:

            if attn_mask.dtype == mindspore.float16:
                attn_score = attn_score - 65500 * ops.einsum("ijbn->bnij", attn_mask)
            else:
                attn_score = attn_score - 1e30 * ops.einsum("ijbn->bnij", attn_mask)


        attn_prob = ops.softmax(attn_score, 3)
        attn_prob = self.dropout(attn_prob)


        if head_mask is not None:
            attn_prob = attn_prob * ops.einsum("ijbn->bnij", head_mask)


        attn_vec = ops.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)

        if output_attentions:
            return attn_vec, ops.einsum("bnij->ijbn", attn_prob)

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing."""
        attn_out = ops.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        attn_out = self.dropout(attn_out)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)

        return output

    def forward(
            self,
            h,
            g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=None,
            target_mapping=None,
            head_mask=None,
            output_attentions=False,
    ):
        """
        Constructs the attention vectors for the XLNetRelativeAttention module.

        Args:
            self (XLNetRelativeAttention): The instance of the XLNetRelativeAttention class.
            h (Tensor): The input tensor h of shape (batch_size, seq_length, hidden_size) representing the hidden states.
            g (Tensor): The input tensor g of shape (batch_size, seq_length, hidden_size) representing the hidden states
                of the global context.
            attn_mask_h (Tensor): The attention mask for h of shape (batch_size, seq_length, seq_length).
            attn_mask_g (Tensor): The attention mask for g of shape (batch_size, seq_length, seq_length).
            r (Tensor): The input tensor r of shape (batch_size, seq_length, hidden_size) representing the hidden states
                of the relative positions.
            seg_mat (Tensor): The input tensor seg_mat of shape (batch_size, seq_length, seq_length) representing the
                segment matrix.
            mems (Tensor, optional): The input tensor mems of shape (mem_len, batch_size, hidden_size) representing
                the memory states. Default is None.
            target_mapping (Tensor, optional): The input tensor target_mapping of shape (batch_size, seq_length, mem_len)
                representing the target mapping. Default is None.
            head_mask (Tensor, optional): The input tensor head_mask of shape (num_heads,) representing the mask for
                the attention heads. Default is None.
            output_attentions (bool, optional): Whether to output the attention probabilities. Default is False.

        Returns:
            Tuple:
                A tuple of two tensors (output_h, output_g) representing the output hidden states for h and g
                respectively. If output_attentions is True, the tuple also contains a tensor attn_prob of shape
                (num_heads, batch_size, seq_length, seq_length) representing the attention probabilities.

        Raises:
            None.
        """
        if g is not None:
            # Two-stream attention with relative positional encoding.
            # content based attention score
            if mems is not None and mems.dim() > 1:
                cat = ops.cat([mems, h], 0)
            else:
                cat = h

            # content-based key head
            k_head_h = ops.einsum("ibh,hnd->ibnd", cat, self.k)

            # content-based value head
            v_head_h = ops.einsum("ibh,hnd->ibnd", cat, self.v)

            # position-based key head
            k_head_r = ops.einsum("ibh,hnd->ibnd", r, self.r)

            # h-stream
            # content-stream query head
            q_head_h = ops.einsum("ibh,hnd->ibnd", h, self.q)

            # core attention ops
            attn_vec_h = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h

            # post processing
            output_h = self.post_attention(h, attn_vec_h)

            # g-stream
            # query-stream query head
            q_head_g = ops.einsum("ibh,hnd->ibnd", g, self.q)

            # core attention ops
            if target_mapping is not None:
                q_head_g = ops.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

                attn_vec_g = ops.einsum("lbnd,mlb->mbnd", attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(
                    q_head_g,
                    k_head_h,
                    v_head_h,
                    k_head_r,
                    seg_mat=seg_mat,
                    attn_mask=attn_mask_g,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                )

                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g

            # post processing
            output_g = self.post_attention(g, attn_vec_g)

            if output_attentions:
                attn_prob = attn_prob_h, attn_prob_g

        else:
            # Multi-head attention with relative positional encoding
            if mems is not None and mems.dim() > 1:
                cat = ops.cat([mems, h], 0)
            else:
                cat = h
            # content heads
            q_head_h = ops.einsum("ibh,hnd->ibnd", h, self.q)
            k_head_h = ops.einsum("ibh,hnd->ibnd", cat, self.k)
            v_head_h = ops.einsum("ibh,hnd->ibnd", cat, self.v)

            # positional heads
            # type casting for fp16 support
            k_head_r = ops.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)

            # core attention ops
            attn_vec = self.rel_attn_core(
                q_head_h,
                k_head_h,
                v_head_h,
                k_head_r,
                seg_mat=seg_mat,
                attn_mask=attn_mask_h,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                attn_vec, attn_prob = attn_vec

            # post processing
            output_h = self.post_attention(h, attn_vec)
            output_g = None

        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class XLNetFeedForward(nn.Module):

    """
    XLNetFeedForward is a class that represents a feed-forward neural network layer for the XLNet model.
    It inherits from nn.Module and contains methods for initializing and forwarding the feed-forward layer.

    The __init__ method initializes the XLNetFeedForward object with the given configuration.
    It sets up the layer normalization, dense layers, dropout, and activation function based on the configuration
    parameters.

    The forward method takes an input tensor and passes it through the feed-forward layer.
    It applies the layer_1, activation function, dropout, layer_2, and layer normalization operations to the input
    tensor, and returns the output tensor after the feed-forward processing.
    """
    def __init__(self, config):
        """
        Initializes an instance of the XLNetFeedForward class.

        Args:
            self: The object instance.
            config: An instance of the configuration class containing model parameters.

        Returns:
            None

        Raises:
            None

        Description:
            This method initializes the XLNetFeedForward object by setting the layer normalization, two dense layers,
            dropout rate, and activation function.

            - self.layer_norm: A LayerNorm module that normalizes the input to the dimensions of the model's hidden size.
            It takes the following parameters:

                - config.d_model: An integer representing the size of the input and output layers.
                - epsilon: A small value added to the variance to avoid division by zero. Default value is
                'config.layer_norm_eps'.

            - self.layer_1: A Dense layer that maps the input to a hidden layer. It takes the following parameters:

                - config.d_model: An integer representing the size of the input layer.
                - config.d_inner: An integer representing the size of the hidden layer.

            - self.layer_2: A Dense layer that maps the hidden layer to the output layer.
            It takes the following parameters:

                - config.d_inner: An integer representing the size of the hidden layer.
                - config.d_model: An integer representing the size of the output layer.

            - self.dropout: A Dropout layer that randomly sets elements to zero during training to prevent overfitting.
            It takes the following parameter:

                - p: The probability of an element to be zeroed. Default value is 'config.dropout'.

            - self.activation_function: The activation function used in the feed-forward layer.
            It can be either a string representing the name of the activation function or a custom activation function.
            If it is a string, it is looked up in the ACT2FN mapping, which maps activation function names to their
            corresponding functions. Otherwise, it is directly assigned to the provided activation function.

        Note:
            - The 'config' parameter should be an instance of the configuration class, which contains necessary model
            parameters.
            - The 'config.ff_activation' parameter can be either a string or a custom activation function.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm([config.d_model], eps=config.layer_norm_eps)
        self.layer_1 = nn.Linear(config.d_model, config.d_inner)
        self.layer_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(p=config.dropout)
        if isinstance(config.ff_activation, str):
            self.activation_function = ACT2FN[config.ff_activation]
        else:
            self.activation_function = config.ff_activation

    def forward(self, inp):
        """
        Constructs the XLNet feed-forward layer.

        Args:
            self (XLNetFeedForward): The instance of the XLNetFeedForward class.
            inp: The input tensor to the feed-forward layer.

        Returns:
            None

        Raises:
            None

        This method applies the XLNet feed-forward layer operations on the input tensor.
        It performs the following steps:

        1. Applies layer_1 on the input tensor.
        2. Applies the activation function on the output of layer_1.
        3. Applies dropout regularization on the output of the activation function.
        4. Applies layer_2 on the output of the dropout operation.
        5. Applies dropout regularization on the output of layer_2.
        6. Adds the input tensor to the output of layer_2 and applies layer normalization.
        7. Returns the final output tensor.

        Note:
            The input tensor is expected to have the shape (batch_size, sequence_length, hidden_size).
        """
        output = inp
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class XLNetLayer(nn.Module):

    """
    Represents a layer of the XLNet model. This class includes methods for initializing the layer,
    forwarding the layer's output, and applying chunking to the forward pass.

    This class inherits from the nn.Module class.

    Attributes:
        rel_attn: XLNetRelativeAttention
            The XLNetRelativeAttention instance for relative attention computation.
        ff: XLNetFeedForward
            The XLNetFeedForward instance for feed-forward computation.
        dropout: nn.Dropout
            The dropout instance for regularization.
        chunk_size_feed_forward: int
            The chunk size for feed-forward computation.
        seq_len_dim: int
            The sequence length dimension.

    Methods:
        __init__:
            Initializes the XLNetLayer with the provided configuration.

        forward:
            Constructs the output of the XLNetLayer based on the provided inputs and optional arguments.

        ff_chunk:
            Applies chunking to the forward pass for the provided output_x.
    """
    def __init__(self, config):
        """
        Initializes an instance of the XLNetLayer class.

        Args:
            self: The instance of the XLNetLayer class.
            config: A configuration object containing parameters for the XLNetLayer initialization.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.rel_attn = XLNetRelativeAttention(config)
        self.ff = XLNetFeedForward(config)
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

    def forward(
            self,
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=None,
            target_mapping=None,
            head_mask=None,
            output_attentions=False,
    ):
        """
        This method forwards the XLNet layer.

        Args:
            self: The instance of the XLNetLayer class.
            output_h (tensor): The output tensor from the previous layer for the current head.
            output_g (tensor): The output tensor from the previous layer for the global context.
            attn_mask_h (tensor): The attention mask for the current head.
            attn_mask_g (tensor): The attention mask for the global context.
            r (int): The number of attention heads.
            seg_mat (tensor): The segment matrix specifying the segment for each token.
            mems (tensor, optional): The memory tensor. Defaults to None.
            target_mapping (tensor, optional): The target mapping tensor. Defaults to None.
            head_mask (tensor, optional): The head mask tensor. Defaults to None.
            output_attentions (bool, optional): Controls whether to output attentions. Defaults to False.

        Returns:
            tuple: A tuple containing the output tensors for the current head and the global context,
                and any additional outputs.

        Raises:
            ValueError: If the dimensions of input tensors are not compatible.
            RuntimeError: If there is a runtime issue during the execution of the method.
            TypeError: If the input types are not as expected.
        """
        outputs = self.rel_attn(
            output_h,
            output_g,
            attn_mask_h,
            attn_mask_g,
            r,
            seg_mat,
            mems=mems,
            target_mapping=target_mapping,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )



        output_h, output_g = outputs[:2]



        if output_g is not None:
            output_g = apply_chunking_to_forward(
                self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_g
            )
        output_h = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, output_h)

        outputs = (output_h, output_g) + outputs[2:]
        return outputs

    def ff_chunk(self, output_x):
        """
        Performs a forward pass through the XLNetLayer for a given input chunk.

        Args:
            self (XLNetLayer): An instance of the XLNetLayer class.
            output_x: The input chunk to be processed. It should be a tensor.

        Returns:
            None.

        Raises:
            None.
        """
        output_x = self.ff(output_x)
        return output_x


class XLNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = XLNetConfig
    base_model_prefix = "transformer"

    def _init_weights(self, cell):
        """Initialize the weights."""
        if isinstance(cell, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            cell.weight.set_data(initializer(Normal(self.config.initializer_range),
                                             cell.weight.shape, cell.weight.dtype))
            if cell.bias !=None:
                cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, nn.Embedding):
            weight = np.random.normal(0.0, self.config.initializer_range, cell.weight.shape)
            if cell.padding_idx:
                weight[cell.padding_idx] = 0
            cell.weight.set_data(Tensor(weight, cell.weight.dtype))
        elif isinstance(cell, nn.LayerNorm):
            cell.weight.set_data(initializer('ones', cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
        elif isinstance(cell, XLNetRelativeAttention):
            for param in [
                cell.q,
                cell.k,
                cell.v,
                cell.o,
                cell.r,
                cell.r_r_bias,
                cell.r_s_bias,
                cell.r_w_bias,
                cell.seg_embed,
            ]:
                param.set_data(initializer(Normal(self.config.initializer_range),
                                           param.shape, param.dtype))
        elif isinstance(cell, XLNetModel):
            cell.mask_emb.set_data(initializer(Normal(self.config.initializer_range),
                                               cell.mask_emb.shape, cell.mask_emb.dtype))


@dataclass
class XLNetModelOutput(ModelOutput):
    """
    Output type of [`XLNetModel`].

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_predict, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    last_hidden_state: mindspore.Tensor
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class XLNetLMHeadModelOutput(ModelOutput):
    """
    Output type of [`XLNetLMHeadModel`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided)
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, num_predict, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).

            `num_predict` corresponds to `target_mapping.shape[1]`. If `target_mapping` is `None`, then `num_predict`
            corresponds to `sequence_length`.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class XLNetForSequenceClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForSequenceClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class XLNetForTokenClassificationOutput(ModelOutput):
    """
    Output type of [`XLNetForTokenClassificationOutput`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class XLNetForMultipleChoiceOutput(ModelOutput):
    """
    Output type of [`XLNetForMultipleChoice`].

    Args:
        loss (`torch.FloatTensor` of shape *(1,)*, *optional*, returned when `labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, num_choices)`):
            *num_choices* is the second dimension of the input tensors. (see *input_ids* above).

            Classification scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    logits: mindspore.Tensor = None
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class XLNetForQuestionAnsweringSimpleOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnsweringSimple`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    start_logits: mindspore.Tensor = None
    end_logits: mindspore.Tensor = None
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


@dataclass
class XLNetForQuestionAnsweringOutput(ModelOutput):
    """
    Output type of [`XLNetForQuestionAnswering`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned if both `start_positions` and `end_positions` are provided):
            Classification loss as the sum of start token, end token (and is_impossible if provided) classification
            losses.
        start_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned
            if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top config.start_n_top start token possibilities (beam-search).
        start_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top)`, *optional*, returned
            if `start_positions` or `end_positions` is not provided):
            Indices for the top config.start_n_top start token possibilities (beam-search).
        end_top_log_probs (`torch.FloatTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`,
            *optional*, returned if `start_positions` or `end_positions` is not provided):
            Log probabilities for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        end_top_index (`torch.LongTensor` of shape `(batch_size, config.start_n_top * config.end_n_top)`, *optional*,
            returned if `start_positions` or `end_positions` is not provided):
            Indices for the top `config.start_n_top * config.end_n_top` end token possibilities (beam-search).
        cls_logits (`torch.FloatTensor` of shape `(batch_size,)`, *optional*, returned if `start_positions` or
            `end_positions` is not provided):
            Log probabilities for the `is_impossible` label of the answers.
        mems (`List[torch.FloatTensor]` of length `config.n_layers`):
            Contains pre-computed hidden-states. Can be used (see `mems` input) to speed up sequential decoding. The
            token ids which have their past given to this model should not be passed as `input_ids` as they have
            already been computed.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[mindspore.Tensor] = None
    start_top_log_probs: Optional[mindspore.Tensor] = None
    start_top_index: Optional[mindspore.Tensor] = None
    end_top_log_probs: Optional[mindspore.Tensor] = None
    end_top_index: Optional[mindspore.Tensor] = None
    cls_logits: Optional[mindspore.Tensor] = None
    mems: Optional[List[mindspore.Tensor]] = None
    hidden_states: Optional[Tuple[mindspore.Tensor, ...]] = None
    attentions: Optional[Tuple[mindspore.Tensor, ...]] = None


class XLNetModel(XLNetPreTrainedModel):

    """
    The XLNetModel class represents a model for XLNet, which is a type of pre-trained model for natural language
    processing. It inherits from the XLNetPreTrainedModel class and provides methods for initializing the model,
    creating attention masks, caching memory, and forwarding the model for inference. The class also includes
    methods for managing input embeddings, positional embeddings, and relative positional encoding.

    The class includes methods for creating attention masks, caching memory, and forwarding the model for inference.
    It also provides functionality for managing input embeddings, positional embeddings, and relative positional
    encoding. The class methods are designed to handle various input parameters and configurations for fine-tuning and
    using the XLNet model for specific NLP tasks. The class is designed to be flexible and efficient for handling
    different use cases and scenarios.
    """
    def __init__(self, config):
        """
        This method initializes an instance of the XLNetModel class with the provided configuration.

        Args:
            self: The instance of the XLNetModel class.
            config:
                A configuration object containing the following parameters:

                - mem_len (int): The length of the memory.
                - reuse_len (int): The length of the segment that can be reused.
                - d_model (int): The dimension of the model.
                - same_length (bool): A flag indicating whether the segments have the same length.
                - attn_type (str): The type of attention mechanism to be used.
                - bi_data (bool): A flag indicating whether the input data is bidirectional.
                - clamp_len (int): The maximum length of the segments.
                - n_layer (int): The number of layers in the model.
                - vocab_size (int): The size of the vocabulary for word embeddings.
                - dropout (float): The dropout rate.

        Returns:
            None.

        Raises:
            ValueError: If the provided configuration is invalid or incomplete.
            TypeError: If the data types of the configuration parameters are not as expected.
            RuntimeError: If an error occurs during the initialization process.
        """
        super().__init__(config)

        self.mem_len = config.mem_len
        self.reuse_len = config.reuse_len
        self.d_model = config.d_model
        self.same_length = config.same_length
        self.attn_type = config.attn_type
        self.bi_data = config.bi_data
        self.clamp_len = config.clamp_len
        self.n_layer = config.n_layer

        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mask_emb = mindspore.Parameter(ops.zeros((1, 1, config.d_model),dtype=mindspore.float32))
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.n_layer)])
        self.dropout = nn.Dropout(p=config.dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        This method retrieves the input embeddings from the XLNetModel.

        Args:
            self (XLNetModel): The instance of the XLNetModel class.
                The self parameter is required to access the word_embedding attribute.

        Returns:
            None.

        Raises:
            None.
        """
        return self.word_embedding

    def set_input_embeddings(self, new_embeddings):
        """
        Method to set new input embeddings for the XLNetModel.

        Args:
            self (XLNetModel): The instance of the XLNetModel class.
                This parameter is a reference to the current XLNetModel instance where the embeddings will be set.
            new_embeddings (any): The new input embeddings to be assigned to the XLNetModel.
                This parameter represents the new embeddings that will replace the existing word embeddings in the model.

        Returns:
            None.

        Raises:
            None.
        """
        self.word_embedding = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        This method `_prune_heads` is a part of the `XLNetModel` class and is used to prune specific heads from the model.

        Args:
            self (XLNetModel): The instance of the XLNetModel class.
            heads_to_prune (int[]): A list of integers representing the indices of the heads to be pruned from the model.
                The indices should be within the valid range of heads for the model.

        Returns:
            None: This method does not return any value.
                It performs the operation in place by pruning the specified heads from the model.

        Raises:
            NotImplementedError: If this method is called directly,
                it raises a NotImplementedError as it should be implemented in a subclass.
        """
        raise NotImplementedError

    def create_mask(self, qlen, mlen):
        """
        Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.

        Args:
            qlen: Sequence length
            mlen: Mask length

        ::

                  same_length=False: same_length=True: <mlen > < qlen > <mlen > < qlen >
               ^ [0 0 0 0 0 1 1 1 1] [0 0 0 0 0 1 1 1 1]
                 [0 0 0 0 0 0 1 1 1] [1 0 0 0 0 0 1 1 1]
            qlen [0 0 0 0 0 0 0 1 1] [1 1 0 0 0 0 0 1 1]
                 [0 0 0 0 0 0 0 0 1] [1 1 1 0 0 0 0 0 1]
               v [0 0 0 0 0 0 0 0 0] [1 1 1 1 0 0 0 0 0]

        """
        mask = ops.ones((qlen, qlen + mlen))
        if self.same_length:
            mask_lo = mask[:, :qlen].tril(-1)
            mask.triu_(mlen + 1)
            mask[:, :qlen] += mask_lo
        else:
            mask.triu_(mlen + 1)

        return mask

    def cache_mem(self, curr_out, prev_mem):
        """
        Caches memory for the XLNetModel.

        Args:
            self (XLNetModel): The instance of the XLNetModel class.
            curr_out (Tensor): The current output tensor.
            prev_mem (Tensor): The previous memory tensor.

        Returns:
            None.

        Raises:
            None.

        """
        # cache hidden states into memory.
        if self.reuse_len is not None and self.reuse_len > 0:
            curr_out = curr_out[: self.reuse_len]

        if self.mem_len is None or self.mem_len == 0:
            # If `use_mems` is active but no `mem_len` is defined, the model behaves like GPT-2 at inference time
            # and returns all of the past and current hidden states.
            cutoff = 0
        else:
            # If `use_mems` is active and `mem_len` is defined, the model returns the last `mem_len` hidden
            # states. This is the preferred setting for training and long-form generation.
            cutoff = -self.mem_len
        if prev_mem is None:
            # if `use_mems` is active and `mem_len` is defined, the model
            new_mem = curr_out[cutoff:]
        else:
            new_mem = ops.cat([prev_mem, curr_out], 0)[cutoff:]

        return new_mem

    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        """
        This method is a static method in the class 'XLNetModel' and is used to generate positional embeddings for input sequences.

        Args:
            pos_seq (torch.Tensor):
                A tensor containing the positional sequence.

                - Type: torch.Tensor
                - Purpose: This tensor represents the positions of the input tokens in the sequence.
                - Restrictions: None

            inv_freq (torch.Tensor):
                A tensor containing the inverse frequency values.

                - Type: torch.Tensor
                - Purpose: This tensor represents the inverse frequency values to be used in the positional
                embedding calculation.
                - Restrictions: None

            bsz (int, optional):
                An optional parameter representing the batch size.

                - Type: int
                - Purpose: This parameter is used to broadcast_to the positional embeddings tensor if provided.
                - Restrictions: None

        Returns:
            torch.Tensor:
                A tensor containing the positional embeddings.

                - Type: torch.Tensor
                - Purpose: This tensor represents the positional embeddings for the input sequence.

        Raises:
            None
        """
        sinusoid_inp = ops.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = ops.cat([ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)], -1)
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = pos_emb.broadcast_to((-1, bsz, -1))

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        """
        Encodes relative positional information for the XLNetModel.

        Args:
            self (XLNetModel): The instance of the XLNetModel class.
            qlen (int): The length of the query sequence.
            klen (int): The length of the key sequence.
            bsz (int, optional): The batch size. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the `attn_type` is not 'bi' or 'uni'.

        """
        # create relative positional encoding.
        freq_seq = ops.arange(0, self.d_model, 2.0, dtype=mindspore.int64).float()
        inv_freq = 1 / ops.pow(10000, (freq_seq / self.d_model))

        if self.attn_type == "bi":
            # beg, end = klen - 1, -qlen
            beg, end = klen, -qlen
        elif self.attn_type == "uni":
            # beg, end = klen - 1, -1
            beg, end = klen, -1
        else:
            raise ValueError(f"Unknown `attn_type` {self.attn_type}.")

        if self.bi_data:
            fwd_pos_seq = ops.arange(beg, end, -1.0, dtype=mindspore.int64).float()
            bwd_pos_seq = ops.arange(-beg, -end, 1.0, dtype=mindspore.int64).float()

            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
                bwd_pos_seq = bwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)

            if bsz is not None:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz // 2)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq, bsz // 2)
            else:
                fwd_pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq)
                bwd_pos_emb = self.positional_embedding(bwd_pos_seq, inv_freq)

            pos_emb = ops.cat([fwd_pos_emb, bwd_pos_emb], 1)
        else:
            fwd_pos_seq = ops.arange(beg, end, -1.0, dtype=mindspore.int64).float()
            if self.clamp_len > 0:
                fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
            pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        return pos_emb

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete after depreciation warning is removed
    ) -> Union[Tuple, XLNetModelOutput]:
        """
        Args:
            self: The instance of the XLNetModel class.
            input_ids (Optional[mindspore.Tensor]): The input tensor containing the token IDs. Default is None.
            attention_mask (Optional[mindspore.Tensor]): The attention mask tensor to avoid attending to padding tokens.
                Default is None.
            mems (Optional[mindspore.Tensor]): The memory tensor for caching previous hidden states. Default is None.
            perm_mask (Optional[mindspore.Tensor]): The permutation mask tensor for partial attention over sequence.
                Default is None.
            target_mapping (Optional[mindspore.Tensor]): The target mapping tensor for masked language modeling.
                Default is None.
            token_type_ids (Optional[mindspore.Tensor]): The tensor containing token type IDs for differentiating
                sequences. Default is None.
            input_mask (Optional[mindspore.Tensor]): The input mask tensor indicating padding tokens. Default is None.
            head_mask (Optional[mindspore.Tensor]): The mask tensor for controlling the attention heads. Default is None.
            inputs_embeds (Optional[mindspore.Tensor]): The tensor containing precomputed embeddings. Default is None.
            use_mems (Optional[bool]): Flag indicating whether to use memory for caching. Default is None.
            output_attentions (Optional[bool]): Flag indicating whether to output attention weights. Default is None.
            output_hidden_states (Optional[bool]): Flag indicating whether to output hidden states of all layers.
                Default is None.
            return_dict (Optional[bool]): Flag indicating whether to return output as a dict. Default is None.

        Returns:
            Union[Tuple, XLNetModelOutput]: The output of the XLNetModel forward method, which includes the last
                hidden state, memory tensors, hidden states of all layers, and attention weights.

        Raises:
            ValueError: Raised if both input_ids and inputs_embeds are specified simultaneously, or if neither
                input_ids nor inputs_embeds are specified.
            FutureWarning: Raised when the 'use_cache' argument is deprecated. Use 'use_mems' instead.
            ValueError: Raised if an unsupported attention type is encountered.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if "use_cache" in kwargs:
            warnings.warn(
                "The `use_cache` argument is deprecated, use `use_mems`"
                " instead.",
                FutureWarning,
            )
            use_mems = kwargs["use_cache"]

        if self.training:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_train
        else:
            use_mems = use_mems if use_mems is not None else self.config.use_mems_eval

        # the original code for XLNet uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_ids = input_ids.swapaxes(1, 0)
            qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.transpose(1, 0, 2)
            qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        token_type_ids = token_type_ids.swapaxes(1, 0) if token_type_ids is not None else None
        input_mask = input_mask.swapaxes(1, 0) if input_mask is not None else None
        attention_mask = attention_mask.swapaxes(1, 0) if attention_mask is not None else None
        perm_mask = perm_mask.permute(1, 2, 0) if perm_mask is not None else None
        target_mapping = target_mapping.permute(1, 2, 0) if target_mapping is not None else None
        mlen = mems[0].shape[0] if mems is not None and mems[0] is not None else 0
        klen = mlen + qlen
        dtype_float = self.dtype
        # Attention mask
        # causal attention mask
        if self.attn_type == "uni":
            attn_mask = self.create_mask(qlen, mlen)
            attn_mask = attn_mask[:, :, None, None]
        elif self.attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError(f"Unsupported attention type: {self.attn_type}")

        # data mask: input mask & perm mask
        assert input_mask is None or attention_mask is None, "You can only use one of input_mask (uses 1 for padding) "
        "or attention_mask (uses 0 for padding, added for compatibility with BERT). Please choose one."
        if input_mask is None and attention_mask is not None:
            input_mask = 1.0 - attention_mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            if mlen > 0:
                mems_mask = ops.zeros((data_mask.shape[0], mlen, bsz)).to(dtype=data_mask.dtype)
                data_mask = ops.cat([mems_mask, data_mask], 1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -ops.eye(qlen).to(dtype=attn_mask.dtype)
            if mlen > 0:
                non_tgt_mask = ops.cat([ops.zeros((qlen, mlen)).to(dtype=attn_mask.dtype), non_tgt_mask],
                                                 -1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(dtype=attn_mask.dtype)
        else:
            non_tgt_mask = None

        # Word embeddings and prepare h & g hidden states
        if inputs_embeds is not None:
            word_emb_k = inputs_embeds
        else:
            word_emb_k = self.word_embedding(input_ids)
        output_h = self.dropout(word_emb_k)
        if target_mapping is not None:
            word_emb_q = self.mask_emb.broadcast_to((target_mapping.shape[0], bsz, -1))
            # else:  # We removed the inp_q input which was same as target mapping
            #     inp_q_ext = inp_q[:, :, None]
            #     word_emb_q = inp_q_ext * self.mask_emb + (1 - inp_q_ext) * word_emb_k
            output_g = self.dropout(word_emb_q)
        else:
            output_g = None
        # Segment embedding
        if token_type_ids is not None:
            # Convert `token_type_ids` to one-hot `seg_mat`
            if mlen > 0:
                mem_pad = ops.zeros((mlen, bsz), dtype=mindspore.int64)
                cat_ids = ops.cat([mem_pad, token_type_ids], 0)
            else:
                cat_ids = token_type_ids
            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = (token_type_ids[:, None] != cat_ids[None, :]).long()
            seg_mat = mindnlp.core.nn.functional.one_hot(seg_mat, 2).to(dtype_float)
        else:
            seg_mat = None

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = self.dropout(pos_emb)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads] (a head_mask for each layer)
        # and head_mask is converted to shape [num_hidden_layers x qlen x klen x bsz x n_head]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                head_mask = head_mask.broadcast_to((self.n_layer, -1, -1, -1, -1))
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            head_mask_dtype = next(iter(self.parameters_dict().items()))[1].dtype
            head_mask = head_mask.to(dtype=head_mask_dtype)
        else:
            head_mask = [None] * self.n_layer

        new_mems = ()
        if mems is None:
            mems = [None] * len(self.layer)
        #r1:new_mems,new_mems=None
        attentions = [] if output_attentions else None
        hidden_states = [] if output_hidden_states else None
        for i, layer_module in enumerate(self.layer):# tot 5
            if use_mems:
                # cache new mems
                new_mems = new_mems + (self.cache_mem(output_h, mems[i]),)
            if output_hidden_states:
                hidden_states.append((output_h, output_g) if output_g is not None else output_h)



            outputs = layer_module(
                output_h,
                output_g,
                attn_mask_h=non_tgt_mask,
                attn_mask_g=attn_mask,
                r=pos_emb,
                seg_mat=seg_mat,
                mems=mems[i],
                target_mapping=target_mapping,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            output_h, output_g = outputs[:2]
            if output_attentions:
                attentions.append(outputs[2])
            #record output for each iter




        # Add last hidden state
        if output_hidden_states:
            hidden_states.append((output_h, output_g) if output_g is not None else output_h)

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we swapaxes back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2)

        if not use_mems:
            new_mems = None

        if output_hidden_states:
            if output_g is not None:
                hidden_states = tuple(h.permute(1, 0, 2) for hs in hidden_states for h in hs)
            else:
                hidden_states = tuple(hs.permute(1, 0, 2) for hs in hidden_states)

        if output_attentions:
            if target_mapping is not None:
                # when target_mapping is provided, there are 2-tuple of attentions
                attentions = tuple(
                    tuple(att_stream.permute(2, 3, 0, 1) for att_stream in t) for t in attentions
                )
            else:
                attentions = tuple(t.permute(2, 3, 0, 1) for t in attentions)

        if not return_dict:
            return tuple(v for v in [output, new_mems, hidden_states, attentions] if v is not None)

        return XLNetModelOutput(
            last_hidden_state=output, mems=new_mems, hidden_states=hidden_states, attentions=attentions
        )
        #r1:output[0,0,:3]=[0.00000000e+00, -5.83709657e-01, -4.92282897e-01]

class XLNetLMHeadModel(XLNetPreTrainedModel):

    """
    A Python class representing the XLNetLMHeadModel, which inherits from XLNetPreTrainedModel.

    XLNetLMHeadModel includes methods for initializing the model, preparing inputs for generation, and forwarding
    the model for language modeling tasks. It also provides a method for reordering the cache during beam search or
    beam sample generation.

    The XLNetLMHeadModel class is designed to work with XLNetModel and nn.Linear to process input data, generate
    predictions, and calculate loss during training.

    The class includes methods for preparing inputs for language generation tasks, such as masked language modeling,
    and for forwarding the model to perform auto-regressive language modeling.

    The _reorder_cache method is used to re-order the mems cache during beam search or beam sample generation to match
    mems with the correct beam_idx at each generation step.

    The class is designed to be used in conjunction with the XLNetModel and XLNetLMHeadModelOutput classes to facilitate
    language modeling tasks.

    For usage examples and additional information, refer to the provided code documentation.
    """
    _tied_weights_keys = ["lm_loss.weight"]

    def __init__(self, config):
        """
        Initializes an instance of the XLNetLMHeadModel class.

        Args:
            self (XLNetLMHeadModel): The instance of the XLNetLMHeadModel class.
            config (XLNetConfig): The configuration object for the XLNet model.

        Returns:
            None.

        Raises:
            ValueError: If the configuration is invalid or missing required attributes.
            TypeError: If the provided config is not of type XLNetConfig.
        """
        super().__init__(config)
        self.attn_type = config.attn_type
        self.same_length = config.same_length

        self.transformer = XLNetModel(config)
        self.lm_loss = nn.Linear(config.d_model, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Returns the output embeddings of the XLNet language model head.

        Args:
            self: An instance of the XLNetLMHeadModel class.

        Returns:
            lm_loss: This method returns the output embeddings of the XLNet language model head.
                The output embeddings are used in various downstream tasks such as text classification
                and named entity recognition.

        Raises:
            None.
        """
        return self.lm_loss

    def set_output_embeddings(self, new_embeddings):
        """
        This method sets the output embeddings for the XLNetLMHeadModel.

        Args:
            self (XLNetLMHeadModel): The instance of the XLNetLMHeadModel class.
            new_embeddings (tensor): The new output embeddings to be set for the model.
                It should be a tensor of the appropriate shape and type.

        Returns:
            None.

        Raises:
            TypeError: If the new_embeddings parameter is not of type tensor.
            ValueError: If the new_embeddings parameter does not meet the required shape or type constraints.
        """
        self.lm_loss = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
    use_mems=True,
    **kwargs):
        '''
        Args:
            self (XLNetLMHeadModel): The instance of the XLNetLMHeadModel class.
            input_ids (Tensor): The input tensor containing tokenized input IDs.
            past_key_values (tuple, optional): A tuple of past key values from previous generation steps.
                Defaults to None.
            use_mems (bool, optional): A boolean flag indicating whether to use memory. Defaults to None.

        Returns:
            None: This method does not return any value.

        Raises:
            ValueError: If the input_ids tensor is not of the expected shape.
            ValueError: If past_key_values are provided but are not in the expected format.
            TypeError: If use_mems is not a boolean value.
        '''
        # Add dummy token at the end (no attention on this one)

        effective_batch_size = input_ids.shape[0]
        dummy_token = ops.zeros((effective_batch_size, 1), dtype=mindspore.int64)

        # At every pass, the attention values for the new token and the two last generated tokens
        # are computed, the rest is reloaded from the `past` cache. A purely auto-regressive model would have
        # offset = 1; offset = 2 seems to have slightly better computation.
        offset = 2

        if past_key_values:
            input_ids = ops.cat([input_ids[:, -offset:], dummy_token], 1)
        else:
            input_ids = ops.cat([input_ids, dummy_token], 1)

        # Build permutation mask so that previous tokens don't see last token
        sequence_length = input_ids.shape[1]
        perm_mask = ops.zeros(
            (effective_batch_size, sequence_length, sequence_length), dtype=mindspore.float32
        )
        perm_mask[:, :, -1] = 1.0

        # We'll only predict the last token
        target_mapping = ops.zeros(
            (effective_batch_size, 1, sequence_length), dtype=mindspore.float32
        )
        target_mapping[:, 0, -1] = 1.0

        inputs = {
            "input_ids": input_ids,
            "perm_mask": perm_mask,
            "target_mapping": target_mapping,
            "use_mems": use_mems,
        }

        # if past is defined in model kwargs then use it for faster decoding
        if past_key_values:
            inputs["mems"] = tuple(layer_past[:-offset, :, :] for layer_past in past_key_values)

        return inputs

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetLMHeadModelOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, num_predict)`, *optional*):
                Labels for masked language modeling. `num_predict` corresponds to `target_mapping.shape[1]`. If
                `target_mapping` is `None`, then `num_predict` corresponds to `sequence_length`.

                The labels should correspond to the masked input words that should be predicted and depends on
                `target_mapping`. Note in order to perform standard auto-regressive language modeling a *<mask>* token has
                to be added to the `input_ids` (see the `prepare_inputs_for_generation` function and examples below)

                Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored, the loss
                is only computed for labels in `[0, ..., config.vocab_size]`

        Returns:
            `Union[Tuple, XLNetLMHeadModelOutput]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, XLNetLMHeadModel
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
            >>> model = XLNetLMHeadModel.from_pretrained("xlnet/xlnet-large-cased")
            ...
            >>> # We show how to setup inputs to predict a next token using a bi-directional context.
            >>> input_ids = torch.tensor(
            ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
            ... ).unsqueeze(
            ...     0
            ... )  # We will predict the masked token
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            >>> target_mapping = torch.zeros(
            ...     (1, 1, input_ids.shape[1]), dtype=torch.float
            ... )  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[
            ...     0, 0, -1
            ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
            ...
            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            >>> next_token_logits = outputs[
            ...     0
            ... ]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
            ...
            >>> # The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
            >>> input_ids = torch.tensor(
            ...     tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)
            ... ).unsqueeze(
            ...     0
            ... )  # We will predict the masked token
            >>> labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
            >>> assert labels.shape[0] == 1, "only one word will be predicted"
            >>> perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            >>> perm_mask[
            ...     :, :, -1
            ... ] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
            >>> target_mapping = torch.zeros(
            ...     (1, 1, input_ids.shape[1]), dtype=torch.float
            ... )  # Shape [1, 1, seq_length] => let's predict one token
            >>> target_mapping[
            ...     0, 0, -1
            ... ] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
            ...
            >>> outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
            >>> loss = outputs.loss
            >>> next_token_logits = (
            ...     outputs.logits
            ... )  # Logits have shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        logits = self.lm_loss(transformer_outputs[0])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            labels = labels.astype(mindspore.int32)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetLMHeadModelOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(mems: List[mindspore.Tensor], beam_idx: mindspore.Tensor) -> List[mindspore.Tensor]:
        """
        This function is used to re-order the `mems` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `mems` with the correct beam_idx at every
        generation step.
        """
        return [layer_past.index_select(1, beam_idx) for layer_past in mems]


class XLNetForSequenceClassification(XLNetPreTrainedModel):

    """
    The `XLNetForSequenceClassification` class is a subclass of `XLNetPreTrainedModel` that represents a model for
    sequence classification tasks using XLNet.

    XLNetForSequenceClassification utilizes the XLNet model architecture combined with a linear layer for classification.
    It can be used for both single-label and multi-label classification tasks.

    To instantiate this class, you need to provide a `config` object as an argument. The `config` object contains various
    configuration parameters for the XLNet model and the classification layer.

    Methods:
        `forward`: This method forwards the XLNetForSequenceClassification model by performing the necessary
            computations. It takes several input tensors, such as `input_ids`, `attention_mask`, `mems`, `perm_mask`,
            `target_mapping`, `token_type_ids`, `input_mask`, `head_mask`, `inputs_embeds`, `labels`, and various
            optional arguments. It returns a tuple of outputs, including `loss`, `logits`, `mems`, `hidden_states`,
            and `attentions`.

    Attributes:
        `num_labels`: The number of labels in the classification task.
        `config`: The configuration object containing parameters for the XLNet model and the classification layer.
        `transformer`: The XLNetModel instance used for sequence representation.
        `sequence_summary`: The SequenceSummary instance used to summarize the sequence representation.
        `logits_proj`: The linear layer used to project the sequence summary to the number of labels.

    Note:
        - The `forward` method automatically determines the `problem_type` based on the `config` parameters and
        the provided `labels`. The `problem_type` can be either 'regression', 'single_label_classification',
        or 'multi_label_classification'.
        - The loss function used for regression is Mean-Square Loss (MSELoss), while for classification, it is
        Cross-Entropy Loss (CrossEntropyLoss) for single-label classification and Binary Cross-Entropy Loss
        (BCEWithLogitsLoss) for multi-label classification.
        - The `forward` method allows for various optional arguments, such as `output_attentions`,
        `output_hidden_states`, and `return_dict`, which control the output format of the XLNet model.
        - The `forward` method returns either a tuple of outputs if `return_dict` is False, or an instance of
        `XLNetForSequenceClassificationOutput` if `return_dict` is True.

    Example:
        ```python
        >>> config = XLNetConfig(...)
        >>> model = XLNetForSequenceClassification(config)
        >>> inputs = {
        ...     'input_ids': input_ids,
        ...     'attention_mask': attention_mask,
        ...     'labels': labels,
        ... }
        >>> outputs = model.forward(**inputs)
        ```
    """
    def __init__(self, config):
        """
        Initializes a new instance of the XLNetForSequenceClassification class.

        Args:
            self: The instance of the XLNetForSequenceClassification class.
            config: An instance of the XLNetConfig class containing the configuration settings for the XLNet model.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForSequenceClassificationOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype in (mindspore.int64, mindspore.int32)):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                labels = labels.astype(mindspore.int32)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLNetForTokenClassification(XLNetPreTrainedModel):

    """
    XLNetForTokenClassification is a class that represents a XLNet model for token classification tasks, inheriting
    from XLNetPreTrainedModel.
    It includes methods for initializing the model with configuration parameters, forwarding the model with various
    input tensors and optional parameters, and computing the token classification loss.

    Attributes:
        num_labels (int): The number of labels for token classification.
        transformer (XLNetModel): The XLNet model for processing input tensors.
        classifier (nn.Linear): The classifier layer for token classification.

    Methods:
        __init__: Initializes the XLNetForTokenClassification instance with the provided configuration.
        forward:
            Constructs the XLNetForTokenClassification model using the input tensors and optional parameters,
            and computes the token classification loss.

            Parameters:

            - input_ids (mindspore.Tensor): The input tensor representing token IDs.
            - attention_mask (mindspore.Tensor): The tensor indicating attention masks.
            - mems (mindspore.Tensor): The tensor for memory inputs.
            - perm_mask (mindspore.Tensor): The tensor for permutation masks.
            - target_mapping (mindspore.Tensor): The tensor for target mapping.
            - token_type_ids (mindspore.Tensor): The tensor for token type IDs.
            - input_mask (mindspore.Tensor): The tensor indicating input masks.
            - head_mask (mindspore.Tensor): The tensor for head masks.
            - inputs_embeds (mindspore.Tensor): The tensor for input embeddings.
            - labels (mindspore.Tensor): The tensor for target labels.
            - use_mems (bool): Flag indicating whether to use memory inputs.
            - output_attentions (bool): Flag indicating whether to output attentions.
            - output_hidden_states (bool): Flag indicating whether to output hidden states.
            - return_dict (bool): Flag indicating whether to return output as a dictionary.
            - kwargs (dict): Additional keyword arguments.

            Returns:

            - Union[Tuple, XLNetForTokenClassificationOutput]: A tuple or XLNetForTokenClassificationOutput object
            containing the computed loss, logits, memories, hidden states, and attentions.

            Notes:

            - labels should be a torch.LongTensor with indices in [0, ..., num_choices].
            - Loss is computed using CrossEntropyLoss.

            Example:
                ```python
                >>> model = XLNetForTokenClassification(config)
                >>> outputs = model.forward(input_ids=input_tensor, attention_mask=attention_mask, labels=label_tensor)
                ```
    """
    def __init__(self, config):
        """
        Initializes an instance of XLNetForTokenClassification.

        Args:
            self (XLNetForTokenClassification): The instance of the XLNetForTokenClassification class.
            config:
                A configuration object containing parameters for the model.

                - Type: Any
                - Purpose: Specifies the configuration settings for the model.
                - Restrictions: Must be compatible with the XLNetModel and nn.Linear classes.

        Returns:
            None.

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
                where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.astype(mindspore.int32)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLNetForMultipleChoice(XLNetPreTrainedModel):

    """
    This class represents an XLNet model for multiple choice tasks. It extends the XLNetPreTrainedModel class and
    provides functionality for forwarding the model and handling multiple choice classification tasks. The class
    includes methods for initializing the model with configuration, forwarding the model with input tensors, and
    computing the loss for multiple choice classification. It utilizes XLNetModel and SequenceSummary modules for
    processing input data and generating model outputs. The class also incorporates various input and output options to
    customize the model behavior during training and evaluation.
    """
    def __init__(self, config):
        """
        Initialize the XLNetForMultipleChoice class.

        Args:
            self (XLNetForMultipleChoice): The instance of the XLNetForMultipleChoice class.
            config: The configuration object containing parameters for model initialization.
               This should be an instance of a configuration class compatible with XLNetModel.

        Returns:
            None.

        Raises:
            TypeError: If the provided config is not of the expected type.
            ValueError: If there are any issues during the initialization process.
        """
        super().__init__(config)

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.logits_proj = nn.Linear(config.d_model, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            labels: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForMultipleChoiceOutput]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
                num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
                `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.shape[-1]) if input_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1]) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.shape[-1]) if attention_mask is not None else None
        flat_input_mask = input_mask.view(-1, input_mask.shape[-1]) if input_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.shape[-2], inputs_embeds.shape[-1])
            if inputs_embeds is not None
            else None
        )

        transformer_outputs = self.transformer(
            flat_input_ids,
            token_type_ids=flat_token_type_ids,
            input_mask=flat_input_mask,
            attention_mask=flat_attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.astype(mindspore.int32)
            loss = loss_fct(reshaped_logits, labels.view(-1))

        if not return_dict:
            output = (reshaped_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForMultipleChoiceOutput(
            loss=loss,
            logits=reshaped_logits,
            mems=transformer_outputs.mems,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class XLNetForQuestionAnsweringSimple(XLNetPreTrainedModel):

    """
    This class represents a simple implementation of the XLNet model for question answering tasks.
    It is designed specifically for question answering tasks where the start and end positions of the answer in
    the input sequence need to be predicted.

    The `XLNetForQuestionAnsweringSimple` class inherits from the `XLNetPreTrainedModel` class, which provides the
    basic infrastructure and functionality for XLNet models.

    The class has a forwardor method `__init__` that initializes the XLNetForQuestionAnsweringSimple instance with
    the given configuration. The configuration includes the number of labels for the classification task and other
    model-specific settings. It also initializes the XLNetModel transformer, which is responsible for the main
    computations of the XLNet model, and the `qa_outputs` module, which is a fully connected layer for predicting start
    and end positions.

    The `forward` method is the main entry point for using the XLNetForQuestionAnsweringSimple model. It takes various
    input tensors, such as `input_ids`, `attention_mask`, and `token_type_ids`, which represent the input sequence and
    its properties. It also takes optional tensors such as `start_positions` and `end_positions`, which are the labels
    for the positions of the start and end of the answer span in the input sequence.

    The method returns either a tuple or a `XLNetForQuestionAnsweringSimpleOutput` object, depending on the `return_dict`
    parameter. The output contains the predicted start and end logits, and optionally, the total loss, the transformer's
    mems, hidden states, and attentions.

    The `forward` method also handles the computation of the loss if the start and end positions are provided.
    It clamps the positions to the length of the sequence and applies the CrossEntropyLoss to calculate the start and
    end losses. The total loss is the average of the start and end losses.

    If the `return_dict` parameter is `False`, the method returns a tuple containing the total loss (if available),
    the start logits, the end logits, and other optional outputs. If the total loss is not available, the tuple contains
    only the logits and optional outputs.

    If the `return_dict` parameter is `True`, the method returns a `XLNetForQuestionAnsweringSimpleOutput` object that
    encapsulates all the outputs.

    Note:
        The class assumes the usage of the `mindspore` library for tensor operations and loss computation.

    """
    def __init__(self, config):
        """
        Initializes an instance of the 'XLNetForQuestionAnsweringSimple' class.

        Args:
            self: The instance of the class.
            config (XLNetConfig): The configuration object for the XLNet model.
                The 'config' object should contain the following attributes:

                - num_labels (int): The number of labels for the classification task.
                This is used to initialize the 'qa_outputs' layer.
                - hidden_size (int): The size of the hidden state in the transformer model.
                This is used to initialize the 'qa_outputs' layer.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            start_positions: Optional[mindspore.Tensor] = None,
            end_positions: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForQuestionAnsweringSimpleOutput]:
        r"""
        Args:
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, -1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            start_positions = start_positions.astype(mindspore.int32)
            end_positions = end_positions.astype(mindspore.int32)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return XLNetForQuestionAnsweringSimpleOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLNetForQuestionAnswering(XLNetPreTrainedModel):

    """
    The XLNetForQuestionAnswering class represents a XLNet model for question answering.
    It inherits from XLNetPreTrainedModel and provides methods for forwarding the model and processing input data for
    question answering tasks. The class includes methods for computing start and end positions of the labelled span,
    determining if a question has an answer or no answer, and computing the plausibility of the answer. Additionally,
    it provides functionality for handling optional masks of tokens that can't be in answers.

    Example:
        ```python
        >>> from transformers import AutoTokenizer, XLNetForQuestionAnswering
        >>> import torch
        ...
        >>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
        >>> model = XLNetForQuestionAnswering.from_pretrained("xlnet/xlnet-base-cased")
        ...
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        >>> start_positions = torch.tensor([1])
        >>> end_positions = torch.tensor([3])
        >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

        loss = outputs.loss
        ```
    """
    def __init__(self, config):
        """
        Initializes an instance of XLNetForQuestionAnswering.

        Args:
            self (XLNetForQuestionAnswering): The instance of the XLNetForQuestionAnswering class.
            config:
                An object containing configuration settings for XLNetForQuestionAnswering.

                - Type: Any
                - Purpose: Specifies the configuration settings to initialize the XLNetForQuestionAnswering instance.
                - Restrictions: Must be a valid configuration object.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__(config)
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.transformer = XLNetModel(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[mindspore.Tensor] = None,
            attention_mask: Optional[mindspore.Tensor] = None,
            mems: Optional[mindspore.Tensor] = None,
            perm_mask: Optional[mindspore.Tensor] = None,
            target_mapping: Optional[mindspore.Tensor] = None,
            token_type_ids: Optional[mindspore.Tensor] = None,
            input_mask: Optional[mindspore.Tensor] = None,
            head_mask: Optional[mindspore.Tensor] = None,
            inputs_embeds: Optional[mindspore.Tensor] = None,
            start_positions: Optional[mindspore.Tensor] = None,
            end_positions: Optional[mindspore.Tensor] = None,
            is_impossible: Optional[mindspore.Tensor] = None,
            cls_index: Optional[mindspore.Tensor] = None,
            p_mask: Optional[mindspore.Tensor] = None,
            use_mems: Optional[bool] = True,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForQuestionAnsweringOutput]:
        r"""
        Args:
            start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the start of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the end of the labelled span for computing the token classification loss.
                Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
                are not taken into account for computing the loss.
            is_impossible (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels whether a question has an answer or no answer (SQuAD 2.0)
            cls_index (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for position (index) of the classification token to use as input for computing plausibility of the
                answer.
            p_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Optional mask of tokens which can't be in answers (e.g. [CLS], [PAD], ...). 1.0 means token should be
                masked. 0.0 mean token is not masked.

        Returns:
            `Union[Tuple, XLNetForQuestionAnsweringOutput]`

        Example:
            ```python
            >>> from transformers import AutoTokenizer, XLNetForQuestionAnswering
            >>> import torch
            ...
            >>> tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased")
            >>> model = XLNetForQuestionAnswering.from_pretrained("xlnet/xlnet-base-cased")
            ...
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
            ...     0
            ... )  # Batch size 1
            >>> start_positions = torch.tensor([1])
            >>> end_positions = torch.tensor([3])
            >>> outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)
            ...
            >>> loss = outputs.loss
            ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        hidden_states = transformer_outputs[0]
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        outputs = transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_positions = start_positions.astype(mindspore.int32)
            end_positions = end_positions.astype(mindspore.int32)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)
                loss_fct_cls = BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5 so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            if not return_dict:
                return (total_loss,) + transformer_outputs[1:]
            else:
                return XLNetForQuestionAnsweringOutput(
                    loss=total_loss,
                    mems=transformer_outputs.mems,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.shape
            start_log_probs = ops.softmax(start_logits, -1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = ops.topk(
                start_log_probs, self.start_n_top, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).broadcast_to((-1, -1, hsz))  # shape (bsz, start_n_top, hsz)
            #start_states = ops.gather_elements(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = ops.gather(hidden_states, -2, start_top_index_exp)
            start_states = start_states.unsqueeze(1).broadcast_to((-1, slen, -1, -1))  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = ops.softmax(end_logits, 1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = ops.topk(
                end_log_probs, self.end_n_top, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
            end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

            start_states = ops.einsum(
                "blh,bl->bh", hidden_states, start_log_probs
            )  # get the representation of START as weighted sum of hidden states
            cls_logits = self.answer_class(
                hidden_states, start_states=start_states, cls_index=cls_index
            )  # Shape (batch size,): one single `cls_logits` for each sample

            if not return_dict:
                outputs = (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)
                return outputs + transformer_outputs[1:]
            else:
                return XLNetForQuestionAnsweringOutput(
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                    mems=transformer_outputs.mems,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )


__all__ = [
    "XLNetModel",
    "XLNetForMultipleChoice",
    "XLNetLMHeadModel",
    "XLNetForQuestionAnswering",
    "XLNetForSequenceClassification",
    "XLNetForTokenClassification",
    "XLNetPreTrainedModel",
    "XLNetForQuestionAnsweringSimple",
]
