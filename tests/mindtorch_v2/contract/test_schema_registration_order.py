import uuid

import pytest

from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_register_kernel_requires_schema_first():
    op_name = f"order_no_schema_{uuid.uuid4().hex}"

    with pytest.raises(RuntimeError, match="schema must be registered"):
        registry.register_kernel(op_name, DispatchKey.CPU, lambda a: a)


def test_register_with_schema_succeeds():
    op_name = f"order_with_schema_{uuid.uuid4().hex}"

    registry.register_schema(op_name, f"{op_name}(Tensor input) -> Tensor")
    registry.register_kernel(op_name, DispatchKey.CPU, lambda a: a)

    entry = registry.get(op_name)
    assert entry.schema_obj is not None
    assert DispatchKey.CPU in entry.kernels
