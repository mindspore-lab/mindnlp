import os

import pytest


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


def _requires_npu(item: pytest.Item) -> bool:
    nodeid = item.nodeid.lower()
    # Match file/function names that are NPU/HCCL/ACL specific.
    return any(token in nodeid for token in ("npu", "hccl", "acl"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _npu_available():
        return

    skip_reason = "NPU-only test skipped in CPU-only environment"
    skip_marker = pytest.mark.skip(reason=skip_reason)
    for item in items:
        if _requires_npu(item):
            item.add_marker(skip_marker)
