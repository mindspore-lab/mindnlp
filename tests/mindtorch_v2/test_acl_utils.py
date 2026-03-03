import types

import numpy as np

import mindtorch_v2._backends.npu.runtime as ascend


def test_host_ptr_from_numpy_uses_bytes_to_ptr(monkeypatch):
    calls = []

    def bytes_to_ptr(data):
        calls.append(data)
        return 123

    dummy_acl = types.SimpleNamespace(util=types.SimpleNamespace(bytes_to_ptr=bytes_to_ptr))
    monkeypatch.setattr(ascend, "acl", dummy_acl)

    arr = np.array([1, 2, 3], dtype=np.float32)
    ptr, buf = ascend._host_ptr_from_numpy(arr)

    assert ptr == 123
    assert calls
    assert isinstance(calls[0], (bytes, bytearray, memoryview))
    assert buf is not None
