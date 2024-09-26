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

from typing import Optional, Union

import mindspore
from mindspore import Tensor
from mindspore.common.initializer import Initializer, Zero

from mindnlp.core import nn, ops
from mindnlp.core.nn import Parameter
from .mappings import _get_rank, _get_group_size


from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)
from .utils import VocabUtility, divide_and_check_no_remainder


class VocabParallelEmbedding(nn.Module):
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
        r"""
        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of the word embeddings.
            padding_idx (Optional[int], optional): The index for padding. Defaults to None.
            init_method (Union[str, Initializer]): The method for initializing the embedding weights. Can be a string representing the method (e.g., 'Zero') or an instance of the Initializer class.
            dtype (mindspore.dtype): The data type of the embedding weights. Defaults to mindspore.float32.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            ValueError: If vocab_size is not a positive integer.
            ValueError: If embedding_size is not a positive integer.
            ValueError: If padding_idx is not None or a positive integer.
            TypeError: If init_method is not a string or an instance of Initializer.
            ValueError: If dtype is not a valid mindspore data type.
        """
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

    def forward(self, input_: Tensor) -> Tensor:  # type: ignore
        r"""
        Constructs a parallel embedding for the given input tensor.
        
        Args:
            self (VocabParallelEmbedding): An instance of the VocabParallelEmbedding class.
            input_ (Tensor): The input tensor to forward the parallel embedding for.
        
        Returns:
            Tensor: A tensor representing the parallel embedding of the input tensor.
        
        Raises:
            None
        
        This method forwards a parallel embedding for the input tensor by performing the following steps:
        
        1. Create an input mask by checking if each element in the input tensor is less than the vocab start index or greater than or equal to the vocab end index.
        2. Subtract the vocab start index from the input tensor to obtain a masked input tensor.
        3. Replace the masked elements in the input tensor with 0 using the input mask.
        4. Get the shape of the masked input tensor.
        5. Gather the embedding weights from the VocabParallelEmbedding instance using the masked input tensor as indices.
        6. Reshape the gathered embedding weights to match the original shape of the masked input tensor.
        7. Fill the masked elements in the gathered embedding weights with 0.0 using the input mask.
        8. Reduce the output tensor from the model parallel region using the output_parallel tensor.
        9. Return the final output tensor.
        
        Note: The vocab start index and vocab end index are properties of the VocabParallelEmbedding class, and the embedding size is also a property of the class.
        """
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


class ParallelEmbedding(nn.Module):
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
        r"""Initialize the ParallelEmbedding class.
        
        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_size (int): The size of each embedding vector.
            padding_idx (Optional[int], optional): The index used for padding. Defaults to None.
            init_method (Union[str, Initializer], optional): The method used for initializing the weight tensor. 
                Can be a string representing the method name or an instance of mindspore.nn.initializer.Initializer.
                Defaults to Zero().
            dtype (mindspore.dtype, optional): The data type of the weight tensor. Defaults to mindspore.float32.
        
        Returns:
            None
        
        Raises:
            None
        """
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

    def forward(self, input_: Tensor) -> Tensor:  # type: ignore
        r"""
        Constructs the parallel embedding for the given input tensor.
        
        Args:
            self (ParallelEmbedding): The instance of the ParallelEmbedding class.
            input_ (Tensor): The input tensor for which the parallel embedding is to be forwarded. It should be a tensor compatible with the model parallel region.
        
        Returns:
            Tensor: The forwarded parallel embedding tensor of type Tensor. The shape and size of the tensor are determined by the input tensor and the embedding size per partition.
        
        Raises:
            ModelParallelRegionError: If the input tensor is not compatible with the model parallel region.
            TensorShapeError: If the shape of the input tensor does not match the expected shape for forwarding the parallel embedding.
            UnsupportedOperationError: If the operation is not supported for the given input tensor or embedding size per partition.
        """
        input_parallel = copy_to_model_parallel_region(input_)
        ori_shape = input_parallel.shape
        output_parallel = ops.gather(self.weight, input_parallel.view(-1), 0).view(
            ori_shape + (self.embedding_size_per_partition, )
        )
        output = gather_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(nn.Module):
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
        r"""
        __init__
        
        Initialize the ColumnParallelLinear class.
        
        Args:
            self: The object itself.
            in_features (int): The size of each input sample.
            out_features (int): The size of each output sample.
            bias (bool, optional): If set to True, a bias term is included. Default is True.
            gather_output (bool, optional): If set to True, the output from all devices will be gathered. Default is True.
            init_method (Union[str, Initializer]): The method used for weight initialization. It can be a string specifying the method or an instance of Initializer. Default is Zero().
            dtype (mindspore.dtype): The data type of the parameters. Default is mindspore.float32.
            stride (int, optional): The stride of the convolution. Default is 1.
            keep_master_weight_for_test (bool, optional): If set to True, master weight will be kept for testing. Default is False.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - TypeError: If in_features, out_features, stride are not integers or if dtype is not a valid mindspore data type.
            - ValueError: If out_features is not divisible by the rank size or if init_method is not a valid initialization method.
            - RuntimeError: If an error occurs during parameter initialization.
        """
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

    def forward(self, input_: Tensor) -> Tensor:  # type: ignore
        r"""
        Constructs the ColumnParallelLinear layer.
        
        Args:
            self (ColumnParallelLinear): An instance of the ColumnParallelLinear class.
            input_ (Tensor): The input tensor to the layer. It must have the shape (batch_size, input_dim).
        
        Returns:
            Tensor: The output tensor of the layer. It will have the shape (batch_size, output_dim).
        
        Raises:
            None.
        """
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


class RowParallelLinear(nn.Module):
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
        r"""
        Initializes a RowParallelLinear object.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Whether to include bias in the linear transformation. Default is True.
            input_is_parallel (bool, optional): Whether the input is parallelized. Default is False.
            init_method (Union[str, Initializer], optional): Initialization method for weights. Default is Zero().
            dtype (mindspore.dtype, optional): Data type of the tensors. Default is mindspore.float32.
            stride (int, optional): Stride value for the linear transformation. Default is 1.
            keep_master_weight_for_test (bool, optional): Whether to keep the master weight for testing. Default is False.
        
        Returns:
            None: This method does not return any value.
        
        Raises:
            - ValueError: If 'in_features' or 'out_features' is not an integer.
            - TypeError: If 'bias', 'input_is_parallel', 'stride', or 'keep_master_weight_for_test' is not a boolean value.
            - TypeError: If 'init_method' is not a string or an Initializer object.
            - ValueError: If 'dtype' is not a valid mindspore data type.
            - ValueError: If 'rank_size' cannot be determined.
            - ValueError: If there is a remainder when dividing 'in_features' by 'rank_size'.
            - ValueError: If the shape of the weight tensor does not match the calculated size per partition.
            - ValueError: If the shape of the bias tensor does not match the number of output features.
        """
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

    def forward(self, input_: Tensor) -> Tensor:  # type:ignore
        r"""
        This method forwards a linear layer operation in a row-parallel fashion.
        
        Args:
            self (RowParallelLinear): The instance of the RowParallelLinear class.
            input_ (Tensor): The input tensor to the linear layer operation. It should be of type Tensor.
        
        Returns:
            Tensor: The output tensor resulting from the linear layer operation.
        
        Raises:
            ValueError: If the input tensor is not of the expected type.
            RuntimeError: If any runtime error occurs during the linear layer operation.
        """
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
