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
"""Tensor Parallel Layers"""

from typing import Callable, Optional
import numpy as np

import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import numpy as mnp

from mindspore.communication import get_rank, get_group_size

from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from .utils import VocabUtility, divide_and_check_no_remainder


def _initialize_affine_weight(
    weight: mindspore.Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable[[mindspore.Tensor], mindspore.Tensor],
    stride: int = 1,
    return_master_weight: bool = False,
) -> Optional[mindspore.Tensor]:
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    # If we only use 1 process for model parallelism, bypass scatter.
    rank_size = get_group_size()
    if rank_size == 1:
        init_method(weight)
        if return_master_weight:
            return weight
        return None

    # Initialize master weight
    master_weight = mnp.empty((out_features, in_features), dtype=weight.dtype)
    init_method(master_weight)

    # Split and copy
    per_partition_per_stride_size = divide_and_check_no_remainder(per_partition_size, stride)
    weight_list = ops.split(master_weight, per_partition_per_stride_size, axis=partition_dim)
    rank_id = get_rank()
    my_weight_list = weight_list[rank_id::rank_size]

    weight = ops.concat(my_weight_list, axis=partition_dim)
    ops.stop_gradient(weight)

    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(nn.Cell):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from mindspore.nn.Embedding and all the default
    values are kept.
    Arguments:
        vocab_size: vocabulary size.
        embedding_size: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        padding_idx: Optional[int] = None,
        init_method: Callable[[mindspore.Tensor], mindspore.Tensor] = mindspore.common.initializer.XavierNormal,
    ) -> None:
        super().__init__()
        # Keep the input dimensions.
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            self.vocab_size, get_rank(), get_group_size()
        )
        self.vocab_size_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights.
        self.weight = mindspore.Parameter(mindspore.Tensor(
            np.random.randn(self.vocab_size_per_partition, self.embedding_size).astype(np.float32)
        ))
        # And initialize.
        _initialize_affine_weight(
            self.weight, self.vocab_size, self.embedding_size, self.vocab_size_per_partition, 0, init_method
        )
        self.embedding = nn.Embedding(
            self.vocab_size_per_partition,
            self.embedding_size,
            padding_idx=self.padding_idx
        )
        for _, param in self.embedding.parameters_and_names():
            param.set_data(self.weight)

    def construct(self, input_: mindspore.Tensor) -> mindspore.Tensor:  # type: ignore
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_ - self.vocab_start_index
        masked_input = ops.masked_fill(masked_input, input_mask, 0)
        # Get the embeddings.
        output_parallel = self.embedding(masked_input)
        # Mask the output embedding.
        output_parallel = ops.masked_fill(output_parallel, input_mask.unsqueeze(-1), 0.0)
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ParallelEmbedding(nn.Cell):
    """Embedding parallelized in the embedding dimension.

    This is mainly adapted from mindspore.nn.Embedding and all the default
    values are kept.
    Arguments:
        vocab_size: vocabulary size.
        embedding_size: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        padding_idx: Optional[int] = None,
        init_method: Callable[[mindspore.Tensor], mindspore.Tensor] = mindspore.common.initializer.XavierNormal
    ) -> None:
        super().__init__()
        # Keep the input dimensions.
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        # Divide the weight matrix along the embedding dimension.
        rank_size = get_group_size()
        self.embedding_size_per_partition = divide_and_check_no_remainder(self.embedding_size, rank_size)

        # Allocate weights.
        self.weight = mindspore.Parameter(mindspore.Tensor(
            np.random.randn(self.vocab_size, self.embedding_size_per_partition).astype(np.float32)
        ))
        # And initialize.
        _initialize_affine_weight(
            self.weight,
            self.vocab_size,
            self.embedding_size,
            self.embedding_size_per_partition,
            1,
            init_method,
            stride=1,
            return_master_weight=False,
        )
        self.embedding = nn.Embedding(
            self.vocab_size,
            self.embedding_size_per_partition,
            padding_idx=self.padding_idx
        )
        for _, param in self.embedding.parameters_and_names():
            param.set_data(self.weight)

    def construct(self, input_: mindspore.Tensor) -> mindspore.Tensor:  # type: ignore
        input_parallel = copy_to_model_parallel_region(input_)
        output_parallel = self.embedding(input_parallel)
        output = gather_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(nn.Cell):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[[mindspore.Tensor], mindspore.Tensor] = mindspore.common.initializer.XavierNormal,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        rank_size = get_group_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, rank_size)

        # Parameters.
        self.weight = mindspore.Parameter(mindspore.Tensor(np.random.randn(self.in_features, self.output_size_per_partition)))
        if bias:
            # Always initialize bias to zero.
            self.bias = mindspore.Parameter(mindspore.Tensor(np.zeros((self.output_size_per_partition), dtype=np.float32)))
        else:
            self.bias = None

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> mindspore.Tensor:
        """get master weight of ColumnParallelLinear"""
        return gather_from_model_parallel_region(self.weight.data).transpose_(0, 1)

    def construct(self, input_: mindspore.Tensor) -> mindspore.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = ops.matmul(input_parallel, self.weight)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class RowParallelLinear(nn.Cell):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[[mindspore.Tensor], mindspore.Tensor] = mindspore.common.initializer.XavierNormal,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        rank_size = get_group_size()
        self.input_size_per_partition = divide_and_check_no_remainder(in_features, rank_size)

        # Parameters.
        # we allocate the transpose.
        self.weight = mindspore.Parameter(mindspore.Tensor(np.random.randn(self.input_size_per_partition, self.out_features)))
        if bias:
            # Always initialize bias to zero.
            self.bias = mindspore.Parameter(mindspore.Tensor(np.zeros((self.out_features), dtype=np.float32)))
        else:
            self.bias = None

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.input_size_per_partition,
            1,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> mindspore.Tensor:
        """get master weight of RowParallelLinear"""
        return gather_from_model_parallel_region(self.weight.data).transpose_(0, 1)

    def construct(self, input_: mindspore.Tensor) -> mindspore.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = ops.matmul(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
