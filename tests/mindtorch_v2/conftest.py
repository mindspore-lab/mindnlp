import multiprocessing
import os
import sys

import pytest

# macOS defaults to "spawn" which cannot pickle locally-defined classes used in
# multi-process DataLoader tests.  Switch to "fork" so that worker processes
# inherit the parent address space (PyTorch's own test suite does the same).
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

_FORCE_CPU_ONLY_ENV = "MINDTORCH_TEST_FORCE_CPU_ONLY"


def _npu_available() -> bool:
    # Optional override for CI/local debugging of CPU-only behavior.
    if os.environ.get(_FORCE_CPU_ONLY_ENV) == "1":
        return False

    try:
        import mindtorch_v2 as torch

        return bool(torch.npu.is_available())
    except Exception:
        return False


def _npu_device_count() -> int:
    # Optional override for CI/local debugging of CPU-only behavior.
    if os.environ.get(_FORCE_CPU_ONLY_ENV) == "1":
        return 0

    try:
        import mindtorch_v2 as torch

        return int(torch.npu.device_count())
    except Exception:
        return 0


def _requires_multicard_npu(item: pytest.Item) -> bool:
    nodeid = item.nodeid.lower()
    return any(token in nodeid for token in ("2card", "multicard"))


def _requires_npu(item: pytest.Item) -> bool:
    nodeid = item.nodeid.lower()
    # Match file/function names that are NPU/HCCL/ACL specific.
    return any(token in nodeid for token in ("npu", "hccl", "acl", "ddp", "pin_memory", "pinned"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _npu_available():
        npu_count = _npu_device_count()
        if npu_count >= 2:
            return

        skip_reason = f"Requires >=2 NPUs, found {npu_count}"
        skip_marker = pytest.mark.skip(reason=skip_reason)
        for item in items:
            if _requires_multicard_npu(item):
                item.add_marker(skip_marker)
        return

    skip_reason = "NPU-only test skipped in CPU-only environment"
    skip_marker = pytest.mark.skip(reason=skip_reason)
    for item in items:
        if _requires_npu(item):
            item.add_marker(skip_marker)
