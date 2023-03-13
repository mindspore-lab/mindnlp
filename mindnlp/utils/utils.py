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

import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import initializer, Normal


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
        self.gamma = Parameter(initializer(Normal(sigma=0.02), (n_in, n_out), mindspore.float32))
        self.beta = Parameter(ops.zeros(n_out))

    def construct(self, x):
        size_out = x.shape[:-1] + (self.n_out,)
        x = ops.addmm(self.beta, x.view(-1, x.shape[-1]), self.gamma)
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
    gama_l = layer.gamma.index_select(axis, index)
    if axis == 0:
        beta_l = layer.beta
    else:
        beta_l = layer.beta[index]
    new_size = list(layer.gamma.shape())
    new_size[axis] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.gamma.requires_grad = False
    new_layer.gamma = gama_l.copy()
    new_layer.gamma.requires_grad = True
    new_layer.beta.requires_grad = False
    new_layer.beta = beta_l.copy()
    new_layer.beta.requires_grad = True
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
