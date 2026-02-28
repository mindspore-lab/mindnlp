"""mindtorch_v2.multiprocessing: multiprocessing wrapper with Tensor/Storage reducers."""

import multiprocessing
import sys

from .reductions import init_reductions


__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]

from multiprocessing import *  # noqa: F403

__all__ += multiprocessing.__all__  # type: ignore[attr-defined]


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    global _sharing_strategy
    if new_strategy not in _all_sharing_strategies:
        raise ValueError(f"unsupported sharing strategy: {new_strategy}")
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    return _sharing_strategy


def get_all_sharing_strategies():
    return set(_all_sharing_strategies)


init_reductions()
