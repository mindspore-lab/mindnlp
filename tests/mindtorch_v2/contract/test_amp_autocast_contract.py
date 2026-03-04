import mindtorch_v2 as torch


def test_autocast_sets_and_restores_state_cpu():
    assert not torch.is_autocast_enabled("cpu")
    prev_dtype = torch.get_autocast_dtype("cpu")

    with torch.amp.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16

    assert not torch.is_autocast_enabled("cpu")
    assert torch.get_autocast_dtype("cpu") == prev_dtype


def test_autocast_nested_enabled_false_subregion_cpu():
    x = torch.randn((4, 4), dtype=torch.float32)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        y = torch.matmul(x, x)
        assert y.dtype == torch.bfloat16

        with torch.amp.autocast("cpu", enabled=False):
            z = torch.matmul(x, x)
            assert z.dtype == torch.float32


def test_autocast_dispatch_key_is_effective_cpu():
    x = torch.randn((4, 4), dtype=torch.float32)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        out = torch.matmul(x, x)
    assert out.dtype == torch.bfloat16


def test_autocast_clears_cache_on_outer_exit(monkeypatch):
    from mindtorch_v2.amp import autocast
    from mindtorch_v2.amp import autocast_mode

    cleared = {"count": 0}

    def _clear_cache():
        cleared["count"] += 1

    monkeypatch.setattr(autocast_mode, "clear_autocast_cache", _clear_cache)

    with autocast("cpu"):
        with autocast("cpu"):
            pass

    assert cleared["count"] == 1


def test_enter_exit_hooks_exist():
    import mindtorch_v2.amp as amp

    assert hasattr(amp, "_enter_autocast")
    assert hasattr(amp, "_exit_autocast")


def test_autocast_invalid_device_type_raises():
    import pytest
    from mindtorch_v2.amp import autocast

    with pytest.raises(RuntimeError):
        with autocast("invalid_device"):
            pass


def test_register_autocast_api_exists():
    import mindtorch_v2.library as library

    assert hasattr(library, "register_autocast")
