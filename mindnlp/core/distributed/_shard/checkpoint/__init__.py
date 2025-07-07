# Keep old package for BC purposes, this file should be removed once
# everything moves to the `core.distributed.checkpoint` package.
import sys
import warnings

from mindnlp import core
from core.distributed.checkpoint import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`core.distributed._shard.checkpoint` will be deprecated, "
        "use `core.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["core.distributed._shard.checkpoint"] = core.distributed.checkpoint
