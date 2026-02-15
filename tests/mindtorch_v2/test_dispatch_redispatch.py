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
