# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindNLP MindSpore Utils"""
# pylint: disable=C0412
# pylint: disable=C0103

import inspect
from typing import Optional

import mindspore
from mindspore import nn, ops, Parameter, Tensor
from mindspore.nn import CrossEntropyLoss
from mindspore.common.initializer import initializer, Normal

from mindnlp.abc import PreTrainedConfig
from mindnlp._legacy.nn import Dropout, Matmul
from ..utils.activations import get_activation

try:
    from mindspore.nn import Identity
except ImportError:
    # Older MindSpore compatibility
    class Identity(nn.Cell):
        r"""A placeholder identity operator that is argument-insensitive."""

        def __init__(self):
            super().__init__()

        def construct(self, hidden_states):
            """
            Return hidden value
            """
            return hidden_states

class Conv1D(nn.Cell):
    """
    1D-convolutional layer Basically works like a linear layer but the weights are transposed.

    Args:
        n_out (`int`): The number of output features.
        n_in (`int`): The number of input features.
    """

    def __init__(self, n_out, n_in):
        super().__init__()
        self.n_out = n_out
        self.weight = Parameter(initializer(Normal(sigma=0.02), (n_in, n_out), mindspore.float32))
        self.bias = Parameter(ops.zeros(n_out, mindspore.float32))
        self.matmul = Matmul()

    def construct(self, x):
        size_out = x.shape[:-1] + (self.n_out,)
        x = self.matmul(x.view(-1, x.shape[-1]), self.weight) + self.bias
        x = x.view(size_out)
        return x


def prune_conv1d_layer(layer, index, axis=1):
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~mindspore_utils.Conv1D`]): The layer to prune.
        index (`mindspore.Tensor[int64]`): The indices to keep in the layer.
        axis (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~mindspore_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    gama_l = layer.weight.index_select(axis, index)
    if axis == 0:
        beta_l = layer.bias
    else:
        beta_l = layer.bias[index]
    new_size = list(layer.weight.shape())
    new_size[axis] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.weight.requires_grad = False
    new_layer.weight = gama_l.copy()
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias = beta_l.copy()
    new_layer.bias.requires_grad = True
    return new_layer


def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], MindSpore.Tensor[int64]]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = ops.ones((n_heads, head_size))
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).eq(1)
    index = ops.arange(len(mask), dtype=mindspore.int64)[mask]
    return heads, index

class SequenceSummary(nn.Cell):
    """
    GPTDoubleHeadsModel and GPT2DoubleHeadsModel class that self.multiple_choice_head
    """

    def __init__(self, config):
        super().__init__()

        self.summary_type = getattr(config, "summary_type", "last")
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Dense(config.hidden_size, num_classes)

        activation_string = getattr(config, "summary_activation", None)
        self.activation = get_activation(activation_string) if activation_string else Identity()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = Dropout(p=config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = Dropout(p=config.summary_last_dropout)

    def construct(self, hidden_states: Tensor, cls_index: Optional[Tensor] = None) -> Tensor:
        if self.summary_type == "last":
            output = hidden_states[:, -1, :]
        elif self.summary_type == "first":
            output = hidden_states[:, 0, :]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = ops.fill(
                    mindspore.int64,
                    hidden_states[..., :1, :].shape,
                    hidden_states.shape[-2] - 1,
                )
            else:
                cls_index = cls_index.expand_dims(-1).expand_dims(-1)
                cls_index = cls_index.expand((-1,) * (cls_index.ndim - 1) + (hidden_states.shape[-1],))
            output = hidden_states.gather_elements(-2, cls_index).squeeze(-2)  # shape (bsz, XX, hidden_size)
        else:
            output = hidden_states

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output

class PoolerStartLogits(nn.Cell):
    """
    Compute SQuAD start logits from sequence hidden states.

    Args:
        config ([`PreTrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, 1)

    def construct(
        self, hidden_states: Tensor,
        p_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            p_mask (`torch.FloatTensor` of shape `(batch_size, seq_len)`, *optional*):
                Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token
                should be masked.

        Returns:
            `torch.FloatTensor`: The start logits for SQuAD.
        """
        x = self.dense(hidden_states).squeeze(-1)

        #TODO : get_parameter_dtype(self)

        # if p_mask is not None:
        #     if get_parameter_dtype(self) == torch.float16:
        #         x = x * (1 - p_mask) - 65500 * p_mask
        #     else:
                # x = x * (1 - p_mask) - 1e30 * p_mask
        x = x * (1 - p_mask) - 1e30 * p_mask
        return x

class SQuADHead(nn.Cell):
    r"""
    A SQuAD head inspired by XLNet.

    Args:
        config ([`PreTrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config):
        super().__init__()
        self.start_n_top = config.start_n_top
        self.end_n_top = config.end_n_top

        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.answer_class = PoolerAnswerClass(config)

    def construct(
        self,
        hidden_states: Tensor,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
        cls_index: Optional[Tensor] = None,
        is_impossible: Optional[Tensor] = None,
        p_mask: Optional[Tensor] = None,
        # return_dict: bool = False,
    ) -> Tensor:

        start_logits = self.start_logits(hidden_states,p_mask=p_mask)   #TODO (hidden_states, p_mask=p_mask)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions, cls_index, is_impossible):
                try:
                    if x is not None and x.ndim > 1:
                        x = ops.squeeze(x,axis=-1)
                except ValueError:
                    pass

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions,p_mask=p_mask)
            #TODO:(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions.astype(mindspore.int32))
            end_loss = loss_fct(end_logits, end_positions.astype(mindspore.int32))
            total_loss = (start_loss + end_loss) / 2

            if cls_index is not None and is_impossible is not None:
                # Predict answerability from the representation of CLS and START
                cls_logits = self.answer_class(hidden_states, start_positions=start_positions, cls_index=cls_index)

                loss_fct_cls = nn.BCEWithLogitsLoss()
                cls_loss = loss_fct_cls(cls_logits, is_impossible)

                # note(zhiliny): by default multiply the loss by 0.5
                # so that the scale is comparable to start_loss and end_loss
                total_loss += cls_loss * 0.5

            return (total_loss,)

        # during inference, compute the end logits based on beam search
        _, slen, hsz = hidden_states.shape   #(bsz,slen.hsz)
        start_log_probs = ops.softmax(start_logits,axis=-1)  # shape (bsz, slen)

        start_top_log_probs, start_top_index = ops.topk(
            start_log_probs, self.start_n_top
        )  # shape (bsz, start_n_top)
        start_top_index_exp = ops.BroadcastTo(shape
                                              =(-1, -1, hsz))(start_top_index.expand_dims(-1))
                                              # shape (bsz, start_n_top, hsz)
        start_states = ops.gather_elements(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
        start_states = ops.BroadcastTo(shape
                                       =(-1, slen, -1, -1))(start_states.expand_dims(1))
                                              # shape (bsz, slen, start_n_top, hsz)

        hidden_states_expanded = hidden_states.expand_dims(2).expand_as(
            start_states
        )  # shape (bsz, slen, start_n_top, hsz)
        p_mask = p_mask.expand_dims(-1) if p_mask is not None else None
        end_logits = self.end_logits(hidden_states_expanded, start_states=start_states)#, p_mask=p_mask
        end_log_probs = ops.softmax(end_logits, axis=1)  # shape (bsz, slen, start_n_top)
        end_top_log_probs, end_top_index = ops.topk(
            end_log_probs, self.end_n_top
        )  # shape (bsz, end_n_top, start_n_top)

        end_top_log_probs = end_top_log_probs.view(-1, self.start_n_top * self.end_n_top)
        end_top_index = end_top_index.view(-1, self.start_n_top * self.end_n_top)

        start_states = ops.matmul(hidden_states,start_log_probs)

        # start_states = ops.einsum("blh,bl->bh", hidden_states, start_log_probs)
        cls_logits = self.answer_class(hidden_states, start_states=start_states, cls_index=cls_index)

        return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)


class PoolerEndLogits(nn.Cell):
    """
    Compute SQuAD end logits from sequence hidden states.

    Args:
        config ([`PreTrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps`
            to use.
    """

    def __init__(self, config: PreTrainedConfig):
        super().__init__()
        self.dense_0 = nn.Dense(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm((config.hidden_size,), epsilon=config.layer_norm_eps)
        self.dense_1 = nn.Dense(config.hidden_size, 1)

    def construct(
        self,
        hidden_states: Tensor,
        start_states: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        p_mask: Optional[Tensor] = None,
    ) -> Tensor:

        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = ops.BroadcastTo(shape
                                              =(-1, -1, hsz))(start_positions[:, None, None])  # shape (bsz, 1, hsz)
            start_states = ops.gather_elements(hidden_states,-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = ops.BroadcastTo(shape=(-1, slen, -1))(start_states)  # shape (bsz, slen, hsz)

        x = self.dense_0(ops.cat([hidden_states, start_states], axis=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)
       #TODO : get_parameter_dtype(self)
        # if p_mask is not None:
        #     if get_parameter_dtype(self) == mindspore.float16:
        #         x = x * (1 - p_mask) - 65500 * p_mask
        #     else:
        #         x = x * (1 - p_mask) - 1e30 * p_mask
        x = x * (1 - p_mask) - 1e30 * p_mask
        return x


class PoolerAnswerClass(nn.Cell):
    """
    Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

    Args:
        config ([`PreTrainedConfig`]):
            The config used by the model, will be used to grab the `hidden_size` of the model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Dense(config.hidden_size * 2, config.hidden_size)
        self.activation = nn.Tanh()
        self.dense_1 = nn.Dense(config.hidden_size, 1, has_bias=False)

    def construct(
        self,
        hidden_states: Tensor,
        start_states: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        cls_index: Optional[Tensor] = None,
    ) -> Tensor:

        # No dependency on end_feature so that we can obtain one single `cls_logits` for each sample.
        hsz = hidden_states.shape[-1]
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            start_positions = ops.BroadcastTo(shape
                                              =(-1, -1, hsz))(start_positions[:, None, None])  # shape (bsz, 1, hsz)
            start_states = ops.gather_elements(hidden_states,-2, start_positions).squeeze(-2)  # shape (bsz, hsz)
        if cls_index is not None:
            cls_index = ops.BroadcastTo(shape=(-1, -1, hsz))(cls_index[:, None, None])  # shape (bsz, 1, hsz)
            cls_token_state = ops.gather_elements(hidden_states,-2, cls_index).squeeze(-2)  # shape (bsz, hsz)
        else:
            cls_token_state = hidden_states[:, -1, :]  # shape (bsz, hsz)

        x = self.dense_0(ops.cat([start_states, cls_token_state], axis=-1))
        x = self.activation(x)
        x = ops.squeeze(self.dense_1(x),axis=-1)

        return x


def prune_linear_layer(layer, index, axis=0):
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (`mindspore.nn.Dense`): The layer to prune.
        index (`mindspore.Tensor[int64]`): The indices to keep in the layer.
        axis (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.
    Returns:
        `mindspore.nn.Dense`: The pruned layer as a new layer with `requires_grad=True`.
    """
    gamma_l = layer.gamma.index_select(axis, index)
    if layer.beta is not None:
        if axis == 1:
            beta_l = layer.beta
        else:
            beta_l = layer.beta[index]
    new_size = list(layer.gamma.shape())
    new_size[axis] = len(index)
    new_layer = nn.Dense(new_size[1], new_size[0], has_bias=layer.beta is not None)
    new_layer.gamma.requires_grad = False
    new_layer.gamma = gamma_l.copy()
    new_layer.gamma.requires_grad = True
    if layer.beta is not None:
        new_layer.beta.requires_grad = False
        new_layer.beta = beta_l.copy()
        new_layer.beta.requires_grad = True
    return new_layer


def apply_chunking_to_forward(forward_fn, chunk_size, chunk_axis, *input_tensors):
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_axis`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.
    Args:
        forward_fn (`Callable[..., mindspore.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_axis (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[mindspore.Tensor]`):
            The input tensors of `forward_fn` which will be chunked
    Returns:
        `mindspore.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    """
    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

     # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_axis]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_axis] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_axis]}"
                )

        if input_tensors[0].shape[chunk_axis] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_axis]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_axis] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, axis=chunk_axis) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return ops.cat(output_chunks, axis=chunk_axis)

    return forward_fn(*input_tensors)
