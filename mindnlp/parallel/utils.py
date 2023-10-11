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
"""Tensor Parallel Utils"""
from typing import Tuple

import mindspore
from mindspore import ops
from mindspore.communication import GlobalComm


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide_and_check_no_remainder(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: mindspore.Tensor, num_partitions: int
) -> Tuple[mindspore.Tensor, ...]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
    """
    # Get the size and dimension.
    last_dim = tensor.ndim - 1
    # Split.
    last_dim_size = divide_and_check_no_remainder(tensor.shape[last_dim], num_partitions)
    tensor_list = ops.split(tensor, last_dim_size, axis=last_dim)

    return tensor_list

def concat_tensor_along_last_dim(tensor, num_partitions):
    """Concat a tensor along its last dimension."""
    last_dim = tensor.ndim - 1
    tensor_list = ops.split(tensor, divide_and_check_no_remainder(tensor.shape[0], num_partitions), axis=0)
    output = ops.concat(tensor_list, axis=last_dim)
    return output


class VocabUtility:
    """Split the vocabulary into `rank_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank_id`
    partition: Note that indices in [first, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank_id: int
    ) -> Tuple[int, int]:
        """get vocab range from vocab size of each partition"""
        index_f = rank_id * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank_id: int, rank_size: int) -> Tuple[int, int]:
        """get vocab range from global vocab size"""
        per_partition_vocab_size = divide_and_check_no_remainder(global_vocab_size, rank_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank_id)

def get_rank(group=GlobalComm.WORLD_COMM_GROUP):
    """get rank"""
    return mindspore.communication.get_rank(group)

def get_group_size(group=GlobalComm.WORLD_COMM_GROUP):
    """get group size"""
    return mindspore.communication.get_group_size(group)
