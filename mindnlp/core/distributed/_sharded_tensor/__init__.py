# Keep old package for BC purposes, this file should be removed once
# everything moves to the `core.distributed._shard` package.
import sys
import warnings

from mindnlp import core
from core.distributed._shard.sharded_tensor import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`core.distributed._sharded_tensor` will be deprecated, "
        "use `core.distributed._shard.sharded_tensor` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules[
    "core.distributed._sharded_tensor"
] = core.distributed._shard.sharded_tensor
