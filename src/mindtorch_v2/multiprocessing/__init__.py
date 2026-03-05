"""mindtorch_v2.multiprocessing: multiprocessing wrapper with Tensor/Storage reducers."""

import multiprocessing
import sys

from .reductions import init_reductions
from .._storage import cleanup_shared_files as _cleanup_shared_files, shared_files_count as _shared_files_count


__all__ = [
    "set_sharing_strategy",
    "get_sharing_strategy",
    "get_all_sharing_strategies",
    "cleanup_shared_files",
    "shared_files_count",
]

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


def cleanup_shared_files():
    _cleanup_shared_files()


def shared_files_count():
    return _shared_files_count()


init_reductions()
