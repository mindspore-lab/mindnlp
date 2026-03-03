from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
from mindtorch_v2._dispatch.registry import registry
from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset


def test_dispatch_fallthrough_skips_key():
    registry.register_schema("aten::ft_op", "ft_op(Tensor a) -> Tensor")

    def cpu_impl(a):
        return a

    registry.register_kernel("aten::ft_op", DispatchKey.CPU, cpu_impl)
    registry.register_fallthrough("aten::ft_op", DispatchKey.Autograd)

    keyset = DispatchKeySet({DispatchKey.Autograd, DispatchKey.CPU})
    out = dispatch_with_keyset("ft_op", keyset, None, 1)
    assert out == 1


def test_dispatch_fallthrough_prefer_next_key():
    registry.register_schema("aten::ft_op2", "ft_op2(Tensor a) -> Tensor")

    def autograd_impl(a):
        return "autograd"

    def cpu_impl(a):
        return "cpu"

    registry.register_kernel("aten::ft_op2", DispatchKey.Autograd, autograd_impl)
    registry.register_kernel("aten::ft_op2", DispatchKey.CPU, cpu_impl)
    registry.register_fallthrough("aten::ft_op2", DispatchKey.Autograd)

    keyset = DispatchKeySet({DispatchKey.Autograd, DispatchKey.CPU})
    out = dispatch_with_keyset("ft_op2", keyset, None, 1)
    assert out == "cpu"
