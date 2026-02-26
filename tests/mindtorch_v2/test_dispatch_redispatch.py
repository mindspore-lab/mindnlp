from mindtorch_v2._dispatch.dispatcher import dispatch, redispatch, current_dispatch_keyset
from mindtorch_v2._dispatch.keys import DispatchKey
from mindtorch_v2._dispatch.registry import registry


def test_redispatch_drops_autograd_key():
    def cpu_impl(a):
        return f"cpu:{a}"

    def autograd_impl(a):
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        return redispatch("test_redispatch", keyset, a)

    registry.register_kernel("test_redispatch", DispatchKey.CPU, cpu_impl)
    registry.register_kernel("test_redispatch", DispatchKey.Autograd, autograd_impl)
    out = dispatch("test_redispatch", "cpu", "x")
    assert out == "cpu:x"



def test_redispatch_drops_backend_specific_autograd_key():
    def cpu_impl(a):
        return f"cpu:{a}"

    def autograd_cpu_impl(a):
        keyset = current_dispatch_keyset().without({DispatchKey.Autograd, DispatchKey.AutogradCPU})
        return redispatch("test_redispatch_backend", keyset, a)

    registry.register_kernel("test_redispatch_backend", DispatchKey.CPU, cpu_impl)
    registry.register_kernel("test_redispatch_backend", DispatchKey.AutogradCPU, autograd_cpu_impl)

    out = dispatch("test_redispatch_backend", "cpu", "x")
    assert out == "cpu:x"



def test_dispatch_prefers_backend_specific_autograd_key_over_generic():
    import mindtorch_v2 as torch

    calls = []

    def cpu_impl(a):
        calls.append("cpu")
        return a

    def autograd_impl(a):
        calls.append("autograd")
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        return redispatch("test_autograd_key_priority", keyset, a)

    def autograd_cpu_impl(a):
        calls.append("autograd_cpu")
        keyset = current_dispatch_keyset().without({DispatchKey.Autograd, DispatchKey.AutogradCPU})
        return redispatch("test_autograd_key_priority", keyset, a)

    registry.register_kernel("test_autograd_key_priority", DispatchKey.CPU, cpu_impl)
    registry.register_kernel("test_autograd_key_priority", DispatchKey.Autograd, autograd_impl)
    registry.register_kernel("test_autograd_key_priority", DispatchKey.AutogradCPU, autograd_cpu_impl)

    x = torch.ones((1,))
    x.requires_grad = True
    dispatch("test_autograd_key_priority", x.device.type, x)

    assert calls == ["autograd_cpu", "cpu"]


def test_dispatch_falls_back_to_generic_autograd_when_backend_missing():
    import mindtorch_v2 as torch

    calls = []

    def cpu_impl(a):
        calls.append("cpu")
        return a

    def autograd_impl(a):
        calls.append("autograd")
        keyset = current_dispatch_keyset().without(DispatchKey.Autograd)
        return redispatch("test_autograd_key_fallback", keyset, a)

    registry.register_kernel("test_autograd_key_fallback", DispatchKey.CPU, cpu_impl)
    registry.register_kernel("test_autograd_key_fallback", DispatchKey.Autograd, autograd_impl)

    x = torch.ones((1,))
    x.requires_grad = True
    dispatch("test_autograd_key_fallback", x.device.type, x)

    assert calls == ["autograd", "cpu"]
