import os
import sys

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(autouse=True)
def _dispatch_registry_isolation():
    from mindtorch_v2._dispatch.registry import registry

    state = registry.snapshot()
    try:
        yield
    finally:
        registry.restore(state)
