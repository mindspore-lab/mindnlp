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
# pylint: disable=W0613
"""Tensor Parallel Layers"""

from typing import Optional, Union

import mindspore
from mindspore import nn, ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Initializer, Zero
from mindspore.ops._tracefunc import trace
from .mappings import _get_rank, _get_group_size


from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from .utils import VocabUtility, divide_and_check_no_remainder


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
        init_method: Union[str, Initializer] = Zero(),
        dtype: mindspore.dtype = mindspore.float32,
    ) -> None:
        super().__init__()
        # Keep the input dimensions.
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        # Divide the weight matrix along the vocaburaly dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.vocab_size, _get_rank(), _get_group_size()
        )
        self.vocab_size_per_partition = self.vocab_end_index - self.vocab_start_index

        # Allocate weights.
        self.weight = Parameter(Tensor(shape=(self.vocab_size_per_partition, self.embedding_size),
                                       dtype=dtype, init=init_method),
                                "weight")

    @trace
    def construct(self, input_: Tensor) -> Tensor:  # type: ignore
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (
            input_ >= self.vocab_end_index
        )
        # Mask the input.
        masked_input = input_ - self.vocab_start_index
        masked_input = ops.masked_fill(masked_input, input_mask, 0)
        # Get the embeddings.
        ori_shape = masked_input.shape
        output_parallel = ops.gather(self.weight, masked_input.view(-1), 0).view(
            ori_shape + (self.embedding_size, )
        )
        # Mask the output embedding.
        output_parallel = ops.masked_fill(
            output_parallel, input_mask.unsqueeze(-1), 0.0
        )
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
        init_method: Union[str, Initializer] = Zero(),
        dtype: mindspore.dtype = mindspore.float32,
    ) -> None:
        super().__init__()
        # Keep the input dimensions.
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        # Divide the weight matrix along the embedding dimension.
        rank_size = _get_group_size()
        self.embedding_size_per_partition = divide_and_check_no_remainder(
            self.embedding_size, rank_size
        )

        # Allocate weights.
        self.weight = Parameter(Tensor(shape=(self.vocab_size, self.embedding_size_per_partition),
                                       dtype=dtype, init=init_method),
                                "weight")

    @trace
    def construct(self, input_: Tensor) -> Tensor:  # type: ignore
        input_parallel = copy_to_model_parallel_region(input_)
        ori_shape = input_parallel.shape
        output_parallel = ops.gather(self.weight, input_parallel.view(-1), 0).view(
            ori_shape + (self.embedding_size_per_partition, )
        )
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
        init_method: Union[str, Initializer] = Zero(),
        dtype: mindspore.dtype = mindspore.float32,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ) -> None:
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        rank_size = _get_group_size()
        self.output_size_per_partition = divide_and_check_no_remainder(
            out_features, rank_size
        )

        # Parameters.
        self.weight = Parameter(Tensor(shape=(self.in_features, self.output_size_per_partition),
                                       dtype=dtype, init=init_method),
                                "weight")
        if bias:
            # Always initialize bias to zero.
            self.bias = Parameter(Tensor(shape=(self.output_size_per_partition,),
                                         dtype=dtype, init=init_method),
                                  "bias")
        else:
            self.bias = None

    def get_master_weight(self) -> Tensor:
        """get master weight of ColumnParallelLinear"""
        return gather_from_model_parallel_region(self.weight).swapaxes(0, 1)

    @trace
    def construct(self, input_: Tensor) -> Tensor:  # type: ignore
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
        init_method: Union[str, Initializer] = Zero(),
        dtype: mindspore.dtype = mindspore.float32,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
    ):
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        rank_size = _get_group_size()
        self.input_size_per_partition = divide_and_check_no_remainder(
            in_features, rank_size
        )

        # Parameters.
        # we allocate the transpose.
        self.weight = Parameter(Tensor(shape=(self.input_size_per_partition, self.out_features),
                                       dtype=dtype, init=init_method),
                                "weight")
        if bias:
            # Always initialize bias to zero.
            self.bias = Parameter(Tensor(shape=(self.out_features,), dtype=dtype, init=init_method), "bias")
        else:
            self.bias = None

    def get_master_weight(self) -> Tensor:
        """get master weight of RowParallelLinear"""
        return gather_from_model_parallel_region(self.weight).swapaxes(0, 1)

    @trace
    def construct(self, input_: Tensor) -> Tensor:  # type:ignore
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


__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
    "ParallelEmbedding",
]
