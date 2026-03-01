import uuid

import pytest

from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry, resolve_dispatch_key
from mindtorch_v2._dispatch.registration import (
    register_autograd_kernels,
    register_forward_kernels,
)


def _new_op(prefix="reg_helper"):
    name = f"{prefix}_{uuid.uuid4().hex}"
    registry.register_schema(name, f"{name}(Tensor input) -> Tensor")
    return name


def test_resolve_dispatch_key_supports_cuda_reserved_dispatch_key():
    assert resolve_dispatch_key("cuda") is DispatchKey.PrivateUse1


def test_register_forward_kernels_registers_cpu_npu_meta_from_one_call():
    op = _new_op("forward_batch")

    def fn(x):
        return x

    register_forward_kernels(op, cpu=fn, npu=fn, meta=fn)

    entry = registry.get(op)
    assert DispatchKey.CPU in entry.kernels
    assert DispatchKey.NPU in entry.kernels
    assert DispatchKey.Meta in entry.kernels


def test_register_forward_kernels_accepts_cuda_key_as_reserved_backend():
    op = _new_op("forward_cuda")

    def fn(x):
        return x

    register_forward_kernels(op, cpu=fn, cuda=fn, meta=fn)

    entry = registry.get(op)
    assert DispatchKey.PrivateUse1 in entry.kernels


def test_register_autograd_kernels_registers_all_autograd_device_keys():
    op = _new_op("autograd_batch")

    def fn(x):
        return x

    register_autograd_kernels(op, default=fn, cpu=fn, npu=fn, meta=fn)

    entry = registry.get(op)
    assert DispatchKey.Autograd in entry.kernels
    assert DispatchKey.AutogradCPU in entry.kernels
    assert DispatchKey.AutogradNPU in entry.kernels
    assert DispatchKey.AutogradMeta in entry.kernels


def test_register_autograd_kernels_accepts_cuda_key_as_reserved_autograd_backend():
    op = _new_op("autograd_cuda")

    def fn(x):
        return x

    register_autograd_kernels(op, default=fn, cpu=fn, cuda=fn, meta=fn)

    entry = registry.get(op)
    assert DispatchKey.AutogradXPU in entry.kernels


def test_register_helpers_still_enforce_schema_first():
    op = f"noschema_{uuid.uuid4().hex}"

    def fn(x):
        return x

    with pytest.raises(RuntimeError, match="schema must be registered"):
        register_forward_kernels(op, cpu=fn)
