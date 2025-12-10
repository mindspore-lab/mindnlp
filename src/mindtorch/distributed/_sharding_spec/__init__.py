# Keep old package for BC purposes, this file should be removed once
# everything moves to the `mindtorch.distributed._shard` package.
import sys
import warnings

import mindtorch
from mindtorch.distributed._shard.sharding_spec import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`mindtorch.distributed._sharding_spec` will be deprecated, "
        "use `mindtorch.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,
    )

import mindtorch.distributed._shard.sharding_spec as _sharding_spec


sys.modules["mindtorch.distributed._sharding_spec"] = _sharding_spec
