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


def test_keyset_includes_autograd_cpu_when_grad_enabled():
    t = torch.ones((2,)).requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.Autograd in keyset
    assert DispatchKey.AutogradCPU in keyset


def test_keyset_includes_autograd_meta_when_grad_enabled():
    t = torch.ones((2,), device="meta").requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.Autograd in keyset
    assert DispatchKey.AutogradMeta in keyset


def test_keyset_includes_autograd_npu_when_grad_enabled():
    if not torch.npu.is_available():
        return
    t = torch.ones((2,), device="npu").requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.Autograd in keyset
    assert DispatchKey.AutogradNPU in keyset


def test_keyset_includes_adinplaceorview_when_grad_enabled():
    t = torch.ones((2,)).requires_grad_()
    keyset = DispatchKeySet.from_tensors((t,), grad_enabled=True)
    assert DispatchKey.ADInplaceOrView in keyset
