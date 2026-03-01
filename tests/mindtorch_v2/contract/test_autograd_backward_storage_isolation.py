from pathlib import Path


def _autograd_backend_source() -> str:
    return Path("src/mindtorch_v2/_backends/autograd.py").read_text(encoding="utf-8")


def test_backward_formulas_do_not_access_storage_payload_directly():
    src = _autograd_backend_source()
    assert "storage().data" not in src
    assert "storage()._data" not in src


def test_target_autograd_npu_kernels_are_not_cpu_only_gated():
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
        marker = f'registry.register_kernel("{op}", DispatchKey.AutogradNPU,'
        section_start = src.find(marker)
        assert section_start != -1, f"missing AutogradNPU registration for {op}"
        section_end = src.find("\nregistry.register_kernel", section_start + 1)
        if section_end == -1:
            section_end = len(src)
        section = src[section_start:section_end]
        assert "cpu_only=True" not in section, f"{op} AutogradNPU should not be cpu_only gated"
