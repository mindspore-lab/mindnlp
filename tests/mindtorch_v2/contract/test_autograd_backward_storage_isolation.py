import re
from pathlib import Path

import mindtorch_v2  # noqa: F401

from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def _autograd_backend_source() -> str:
    return Path("src/mindtorch_v2/_backends/autograd.py").read_text(encoding="utf-8")


def test_backward_formulas_do_not_access_storage_payload_directly():
    src = _autograd_backend_source()
    assert "storage().data" not in src
    assert "storage()._data" not in src


def test_target_autograd_npu_kernels_are_registered_and_not_cpu_only_config():
    src = _autograd_backend_source()
    target_ops = (
        "relu",
        "relu_",
        "abs",
        "neg",
        "silu",
        "leaky_relu",
        "elu",
        "mish",
        "prelu",
    )

    for op in target_ops:
        cpu_only_pattern = (
            rf'_autograd_(?:unary|unary_args|inplace)\("{re.escape(op)}"[^\n]*cpu_only=True'
        )
        assert re.search(cpu_only_pattern, src) is None, (
            f"{op} should not use cpu_only=True in autograd wrapper config"
        )

        entry = registry.get(f"aten::{op}")
        assert DispatchKey.AutogradNPU in entry.kernels, (
            f"missing AutogradNPU registration for {op}"
        )
