from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet


def test_priority_order_pipeline_after_backendselect():
    keyset = DispatchKeySet.from_mask(
        DispatchKey.BackendSelect | DispatchKey.Pipeline | DispatchKey.CPU
    )
    order = [k for k in keyset.iter_keys()]
    assert order[0] == DispatchKey.BackendSelect
    assert order[1] == DispatchKey.Pipeline
