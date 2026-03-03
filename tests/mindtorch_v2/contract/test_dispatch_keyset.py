from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet


def test_priority_order_pipeline_after_backendselect():
    keyset = DispatchKeySet.from_mask(
        DispatchKey.BackendSelect | DispatchKey.Pipeline | DispatchKey.CPU
    )
    order = [k for k in keyset.iter_keys()]
    assert order[0] == DispatchKey.BackendSelect
    assert order[1] == DispatchKey.Pipeline
from mindtorch_v2._dispatch.keys import include_keys, exclude_keys
from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset
from mindtorch_v2._dispatch.registry import registry


def test_tls_exclude_applies_to_dispatch_and_redispatch():
    calls = []

    def cpu_kernel(x):
        calls.append("cpu")
        return x

    def meta_kernel(x):
        calls.append("meta")
        return x

    registry.register_kernel("aten::dummy_dispatch", DispatchKey.CPU, cpu_kernel)
    registry.register_kernel("aten::dummy_dispatch", DispatchKey.Meta, meta_kernel)

    keyset = DispatchKeySet.from_mask(DispatchKey.CPU | DispatchKey.Meta)
    with exclude_keys(DispatchKey.Meta):
        dispatch_with_keyset("dummy_dispatch", keyset, None, 1)
    assert calls == ["cpu"]

def test_composite_keys_fallthrough():
    calls = []

    def cpu_kernel(x):
        calls.append("cpu")
        return x

    registry.register_kernel("aten::dummy_composite", DispatchKey.CPU, cpu_kernel)

    keyset = DispatchKeySet.from_mask(
        DispatchKey.CompositeImplicitAutograd | DispatchKey.CPU
    )
    dispatch_with_keyset("dummy_composite", keyset, None, 1)
    assert calls == ["cpu"]
