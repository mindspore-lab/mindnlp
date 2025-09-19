# Keep old package for BC purposes, this file should be removed once
# everything moves to the `mindtorch.distributed.checkpoint` package.
import sys
import warnings

import mindtorch
from mindtorch.distributed.checkpoint import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`mindtorch.distributed._shard.checkpoint` will be deprecated, "
        "use `mindtorch.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["mindtorch.distributed._shard.checkpoint"] = mindtorch.distributed.checkpoint
