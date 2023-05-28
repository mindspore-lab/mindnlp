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

from mindnlp._legacy.ops import AllGather
from .utils import divide_and_check_no_remainder, split_tensor_along_last_dim


def _reduce(input_: mindspore.Tensor) -> mindspore.Tensor:
    """All-reduce the the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_group_size() == 1:
        return input_

    # All-reduce.
    _all_reduce = ops.AllReduce()
    _all_reduce(input_)

    return input_

def _split(input_: mindspore.Tensor) -> mindspore.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    # Bypass the function if we are using only 1 GPU.
    if get_group_size() == 1:
        return input_

    # Split along last dimension.
    rank_size = get_group_size()
    input_list = split_tensor_along_last_dim(input_, rank_size)

    rank = get_rank()
    output = input_list[rank]

    return output

def _gather(input_: mindspore.Tensor) -> mindspore.Tensor:
    """Gather tensors and concatinate along the last dimension."""
    # Bypass the function if we are using only 1 GPU.
    rank_size = get_group_size()
    if rank_size == 1:
        return input_

    # # Size and dimension.
    last_dim = input_.ndim - 1

    _all_gather = AllGather()
    tensor = _all_gather(input_)
    tensor_list = ops.split(tensor, divide_and_check_no_remainder(tensor.shape[0], rank_size), axis=0)
    output = ops.concat(tensor_list, axis=last_dim)

    return output

class _CopyToModelParallelRegion(nn.Cell):
    """Pass the input to the model parallel region."""
    def construct(self, input_):
        return input_

    def bprop(self, input_, out, dout):
        """_CopyToModelParallelRegion backward method"""
        return (_reduce(dout),)


class _ReduceFromModelParallelRegion(nn.Cell):
    """All-redcue the input from the model parallel region."""
    def construct(self, input_):
        return _reduce(input_)

    def bprop(self, input_, out, dout):
        """_ReduceFromModelParallelRegion backward method"""
        return (dout, )


class _ScatterToModelParallelRegion(nn.Cell):
    """Split the input and keep only the corresponding chuck to the rank."""
    def construct(self, input_):
        return _split(input_)

    def bprop(self, input_, out, dout):
        """_ScatterToModelParallelRegion backward method"""
        return (_gather(dout),)

class _GatherFromModelParallelRegion(nn.Cell):
    """Gather the input from model parallel region and concatinate."""

    def construct(self, input_):
        return _gather(input_)

    def bprop(self, input_, out, dout):
        """_GatherFromModelParallelRegion backward method"""
        return (_split(dout),)

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
