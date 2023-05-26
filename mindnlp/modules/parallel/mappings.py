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
"""Tensor Parallel mappings"""
# pylint: disable=unused-argument

import mindspore
from mindspore import nn
from mindspore import ops

from mindspore.communication import get_rank, get_group_size

from .utils import divide_and_check_no_remainder, split_tensor_along_last_dim


class _CopyToModelParallelRegion(nn.Cell):
    """Pass the input to the model parallel region."""
    def __init__(self):
        super().__init__()
        self.rank_size = get_group_size()
        self.all_reduce = ops.AllReduce()

    def construct(self, input_):  # type: ignore
        return input_

    def bprop(self, input_, out, dout):  # type: ignore
        """_CopyToModelParallelRegion backward method"""
        return (self._reduce(dout), )

    def _reduce(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """All-reduce the the input tensor across model parallel group."""
        # Bypass the function if we are using only 1 GPU.
        if self.rank_size == 1:
            return input_

        # All-reduce.
        self.all_reduce(input_)

        return input_


class _ReduceFromModelParallelRegion(nn.Cell):
    """All-redcue the input from the model parallel region."""
    def __init__(self):
        super().__init__()
        self.rank_size = get_group_size()
        self.all_reduce = ops.AllReduce(ops.ReduceOp.SUM)

    def construct(self, input_):  # type: ignore
        return self._reduce(input_)

    def bprop(self, input_, out, dout):  # type: ignore
        """_ReduceFromModelParallelRegion backward method"""
        return (dout, )

    def _reduce(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """All-reduce the the input tensor across model parallel group."""
        # Bypass the function if we are using only 1 GPU.
        if self.rank_size == 1:
            return input_

        # All-reduce.
        output = self.all_reduce(input_)

        return output


class _ScatterToModelParallelRegion(nn.Cell):
    """Split the input and keep only the corresponding chuck to the rank."""
    def __init__(self):
        super().__init__()
        self.rank_id = get_rank()
        self.rank_size = get_group_size()
        self.all_gather = ops.AllGather()

    def construct(self, input_):  # type: ignore
        return self._split(input_)

    def bprop(self, input_, out, dout):  # type: ignore
        """_ScatterToModelParallelRegion backward method"""
        return (self._gather(dout), )

    def _split(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """Split the tensor along its last dimension and keep the
        corresponding slice."""
        # Bypass the function if we are using only 1 GPU.
        if  self.rank_size == 1:
            return input_

        # Split along last dimension.
        input_list = split_tensor_along_last_dim(input_, self.rank_size)

        # Note: torch.split does not create contiguous tensors by default.
        output = input_list[self.rank_id]

        return output

    def _gather(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """Gather tensors and concatinate along the last dimension."""
        # Bypass the function if we are using only 1 GPU.
        if self.rank_size  == 1:
            return input_

        # Size and dimension.
        last_dim = input_.ndim - 1

        tensor = self.all_gather(input_)
        tensor_list = ops.split(tensor, divide_and_check_no_remainder(tensor.shape[0], self.rank_size), axis=0)
        output = ops.concat(tensor_list, axis=last_dim)

        return output


class _GatherFromModelParallelRegion(nn.Cell):
    """Gather the input from model parallel region and concatinate."""
    def __init__(self):
        super().__init__()
        self.rank_id = get_rank()
        self.rank_size = get_group_size()
        self.all_gather = ops.AllGather()

    def construct(self, input_):  # type: ignore
        return self._gather(input_)

    def bprop(self, input_, out, dout):  # type: ignore
        """_GatherFromModelParallelRegion backward method"""
        return (self._split(dout), )

    def _gather(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """Gather tensors and concatinate along the last dimension."""
        # Bypass the function if we are using only 1 GPU.
        if self.rank_size  == 1:
            return input_

        # Size and dimension.
        last_dim = input_.ndim - 1

        tensor = self.all_gather(input_)
        tensor_list = ops.split(tensor, divide_and_check_no_remainder(tensor.shape[0], self.rank_size), axis=0)
        output = ops.concat(tensor_list, axis=last_dim)

        return output

    def _split(self, input_: mindspore.Tensor) -> mindspore.Tensor:
        """Split the tensor along its last dimension and keep the
        corresponding slice."""
        # Bypass the function if we are using only 1 GPU.
        if  self.rank_size == 1:
            return input_

        # Split along last dimension.
        input_list = split_tensor_along_last_dim(input_, self.rank_size)

        # Note: torch.split does not create contiguous tensors by default.
        output = input_list[self.rank_id]

        return output


_copyToModel = _CopyToModelParallelRegion()
_reduceFromModel = _ReduceFromModelParallelRegion()
_scatterToModel = _ScatterToModelParallelRegion()
_gatherFromModel = _GatherFromModelParallelRegion()


def copy_to_model_parallel_region(input_: mindspore.Tensor) -> mindspore.Tensor:
    """copy to model parallel region"""
    return _copyToModel(input_)


def reduce_from_model_parallel_region(input_: mindspore.Tensor) -> mindspore.Tensor:
    """reduce from model parallel region"""
    return _reduceFromModel(input_)


def scatter_to_model_parallel_region(input_: mindspore.Tensor) -> mindspore.Tensor:
    """scatter to model parallel region"""
    return _scatterToModel(input_)


def gather_from_model_parallel_region(input_: mindspore.Tensor) -> mindspore.Tensor:
    """gather from model parallel region"""
    return _gatherFromModel(input_)
