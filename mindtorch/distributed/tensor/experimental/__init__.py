# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

from mindtorch.distributed.tensor._api import DTensor
from mindtorch.distributed.tensor.experimental._attention import context_parallel
from mindtorch.distributed.tensor.experimental._func_map import local_map
from mindtorch.distributed.tensor.experimental._register_sharding import register_sharding


__all__ = ["context_parallel", "implicit_replication", "local_map", "register_sharding"]


@contextmanager
def implicit_replication() -> Iterator[None]:
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``mindtorch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``mindtorch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False


# Set namespace for exposed private names
context_parallel.__module__ = "mindtorch.distributed.tensor.experimental"
implicit_replication.__module__ = "mindtorch.distributed.tensor.experimental"
local_map.__module__ = "mindtorch.distributed.tensor.experimental"
register_sharding.__module__ = "mindtorch.distributed.tensor.experimental"
