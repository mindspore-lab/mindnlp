# Keep old package for BC purposes, this file should be removed once
# everything moves to the `core.distributed._shard` package.
import sys
import warnings

from mindnlp import core
from core.distributed._shard.sharding_spec import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`core.distributed._sharding_spec` will be deprecated, "
        "use `core.distributed._shard.sharding_spec` instead",
        DeprecationWarning,
        stacklevel=2,
    )

from mindnlp import core.distributed._shard.sharding_spec as _sharding_spec


sys.modules["core.distributed._sharding_spec"] = _sharding_spec
