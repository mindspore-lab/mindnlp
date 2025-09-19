# Keep old package for BC purposes, this file should be removed once
# everything moves to the `mindtorch.distributed._shard` package.
import sys
import warnings

import mindtorch
from mindtorch.distributed._shard.sharded_tensor import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`mindtorch.distributed._sharded_tensor` will be deprecated, "
        "use `mindtorch.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules[
    "mindtorch.distributed._sharded_tensor"
] = mindtorch.distributed._shard.sharded_tensor
