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


def test_autocast_unavailable_device_raises():
    import pytest
    from mindtorch_v2.amp import autocast
    from mindtorch_v2 import npu as npu_api

    if npu_api.is_available():
        pytest.skip("npu available")
    with pytest.raises(RuntimeError):
        with autocast("npu"):
            pass


def test_autocast_unsupported_dtype_warns_and_disables():
    import pytest
    from mindtorch_v2.amp import autocast

    with pytest.warns(UserWarning):
        with autocast("cpu", dtype=torch.float32) as mode:
            assert not mode._enabled


def test_top_level_autocast_cache_api_shape_matches_torch():
    import pytest

    # torch.is_autocast_cache_enabled() takes no args.
    with pytest.raises(TypeError):
        torch.is_autocast_cache_enabled("cpu")

    # torch.set_autocast_cache_enabled() takes exactly one bool arg.
    with pytest.raises(TypeError):
        torch.set_autocast_cache_enabled("cpu", True)





def _define_identity_custom_op():
    import uuid
    import mindtorch_v2.library as library

    ns = f"ampautocast_{uuid.uuid4().hex}"
    lib = library.Library(ns, "DEF")
    lib.define("identity(Tensor x) -> Tensor")

    @lib.impl("identity", dispatch_key="CPU")
    def _identity_cpu(x):
        return x

    return f"{ns}::identity"


def test_register_autocast_raises_for_unknown_op():
    import pytest
    import mindtorch_v2.library as library

    with pytest.raises(RuntimeError):
        library.register_autocast("ampautocast_missing::identity", "cpu", torch.bfloat16)


def test_register_autocast_rejects_invalid_args_like_torch():
    import pytest
    import mindtorch_v2.library as library

    with pytest.raises(ValueError):
        library.register_autocast(1, "cpu", torch.bfloat16)

    with pytest.raises(ValueError):
        library.register_autocast("aten::add", "npu", torch.bfloat16)

    with pytest.raises(RuntimeError):
        library.register_autocast("aten::add", "cpu", torch.bfloat16)




def test_register_autocast_duplicate_registration_raises_like_torch():
    import pytest
    import mindtorch_v2.library as library

    qualname = _define_identity_custom_op()
    library.register_autocast(qualname, "cpu", torch.bfloat16)

    with pytest.raises(RuntimeError):
        library.register_autocast(qualname, "cpu", torch.float16)

def test_register_autocast_affects_custom_op_dispatch_in_autocast_region():
    import mindtorch_v2.library as library

    qualname = _define_identity_custom_op()
    x = torch.randn((4, 4), dtype=torch.float32)

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        before = torch._dispatch.dispatch(qualname, x.device, x)
    assert before.dtype == torch.float32

    result = library.register_autocast(qualname, "cpu", torch.bfloat16)
    assert result is None

    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        after = torch._dispatch.dispatch(qualname, x.device, x)
    assert after.dtype == torch.bfloat16

def test_register_autocast_api_exists():
    import mindtorch_v2.library as library

    assert hasattr(library, "register_autocast")


def test_get_autocast_dtype_requires_device_type_like_torch():
    import pytest

    with pytest.raises(TypeError):
        torch.get_autocast_dtype()


def test_set_autocast_enabled_accepts_single_arg_like_torch():
    from mindtorch_v2.amp import state as amp_state

    default_device = getattr(amp_state, "_DEFAULT_DEVICE", "cpu")
    prev_cpu = torch.is_autocast_enabled("cpu")
    prev_default_device = torch.is_autocast_enabled(default_device)
    prev_default = torch.is_autocast_enabled()

    # torch accepts one-arg set_autocast_enabled(enabled), affecting default backend state.
    torch.set_autocast_enabled(True)
    assert torch.is_autocast_enabled() is True
    assert torch.is_autocast_enabled(default_device) is True
    if default_device != "cpu":
        assert torch.is_autocast_enabled("cpu") == prev_cpu

    torch.set_autocast_enabled(False)
    assert torch.is_autocast_enabled() is False
    assert torch.is_autocast_enabled(default_device) is False
    if default_device != "cpu":
        assert torch.is_autocast_enabled("cpu") == prev_cpu

    # restore default state for test isolation
    torch.set_autocast_enabled(prev_default)
    torch.set_autocast_enabled(default_device, prev_default_device)


def test_get_autocast_dtype_invalid_device_raises_runtimeerror_like_torch():
    import pytest

    with pytest.raises(RuntimeError):
        torch.get_autocast_dtype("invalid_device")


def test_set_autocast_enabled_invalid_device_raises_runtimeerror_like_torch():
    import pytest

    with pytest.raises(RuntimeError):
        torch.set_autocast_enabled("invalid_device", True)


def test_set_autocast_enabled_validates_enabled_type_like_torch():
    import pytest

    with pytest.raises(TypeError):
        torch.set_autocast_enabled("cpu", 1)

    with pytest.raises(TypeError):
        torch.set_autocast_enabled("cpu", None)

    with pytest.raises(TypeError):
        torch.set_autocast_enabled(1)


def test_set_autocast_dtype_validates_args_like_torch():
    import pytest

    with pytest.raises(TypeError):
        torch.set_autocast_dtype(None, torch.bfloat16)

    with pytest.raises(TypeError):
        torch.set_autocast_dtype("cpu", None)

    with pytest.raises(TypeError):
        torch.set_autocast_dtype("cpu", 1)
