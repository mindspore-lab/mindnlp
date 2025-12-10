# Copyright (c) Meta Platforms, Inc. and affiliates

import mindtorch
# import mindtorch.distributed.tensor._ops  # force import all built-in dtensor ops
from mindtorch.distributed.device_mesh import DeviceMesh, init_device_mesh  # noqa: F401
from mindtorch.distributed.tensor._api import (
    distribute_module,
    distribute_tensor,
    DTensor,
    empty,
    full,
    ones,
    rand,
    randn,
    zeros,
)
from mindtorch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from mindtorch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
# from mindtorch.utils._foreach_utils import (
#     _foreach_supported_types as _util_foreach_supported_types,
# )


# All public APIs from dtensor package
__all__ = [
    "DTensor",
    "distribute_tensor",
    "distribute_module",
    "Shard",
    "Replicate",
    "Partial",
    "Placement",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
]

# For weights_only mindtorch.load
from ._dtensor_spec import DTensorSpec as _DTensorSpec, TensorMeta as _TensorMeta


# mindtorch.serialization.add_safe_globals(
#     [
#         DeviceMesh,
#         _DTensorSpec,
#         _TensorMeta,
#         DTensor,
#         Partial,
#         Replicate,
#         Shard,
#     ]
# )


# Append DTensor to the list of supported types for foreach implementation for optimizer
# and clip_grad_norm_ so that we will try to use foreach over the for-loop implementation on CUDA.
if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(DTensor)

# if DTensor not in _util_foreach_supported_types:
#     _util_foreach_supported_types.append(DTensor)  # type: ignore[arg-type]


# Set namespace for exposed private names
DTensor.__module__ = "mindtorch.distributed.tensor"
distribute_tensor.__module__ = "mindtorch.distributed.tensor"
distribute_module.__module__ = "mindtorch.distributed.tensor"
ones.__module__ = "mindtorch.distributed.tensor"
empty.__module__ = "mindtorch.distributed.tensor"
full.__module__ = "mindtorch.distributed.tensor"
rand.__module__ = "mindtorch.distributed.tensor"
randn.__module__ = "mindtorch.distributed.tensor"
zeros.__module__ = "mindtorch.distributed.tensor"
