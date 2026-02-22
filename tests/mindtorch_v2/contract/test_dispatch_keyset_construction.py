from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
import mindtorch_v2 as torch


def test_keyset_excludes_placeholders_by_default():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,))
    assert DispatchKey.BackendSelect not in keyset
    assert DispatchKey.AutogradNPU not in keyset
    assert DispatchKey.Autocast not in keyset


def test_keyset_includes_pipeline_only_when_enabled():
    t = torch.ones((2,))
    keyset = DispatchKeySet.from_tensors((t,), pipeline_enabled=False)
    assert DispatchKey.Pipeline not in keyset
    keyset = DispatchKeySet.from_tensors((t,), pipeline_enabled=True)
    assert DispatchKey.Pipeline in keyset
