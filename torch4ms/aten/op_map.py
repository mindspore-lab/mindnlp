# Copyright 2025 Google LLC
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

"""Torch ops implemented using jax."""

import functools

import math
from mindspore import ops, mint
import functools
import torch

from .op_register import register_torch_dispatch_op

# Keys are OpOverload, value is a callable that takes
# Tensor
all_ops = {}


def op(*aten, **kwargs):
    def inner(func):
        for a in aten:
            register_torch_dispatch_op(a, func, **kwargs)
        return func

    return inner


@op(
    torch.ops.aten.view_copy,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten.reshape,
)
def _aten_unsafe_view(x, shape):
    return mint.reshape(x, shape)


@op(torch.ops.aten.add.Tensor)
@op(torch.ops.aten.add.Scalar)
def _aten_add(x, y, *, alpha=1):
    """if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):

    assert x.dtype == y.dtype, (x.dtype, y.dtype)
    """
    res = mint.add(x, y, alpha=alpha)
    return res
