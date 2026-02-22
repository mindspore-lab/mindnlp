from mindtorch_v2._dispatch.keys import DispatchKey, DispatchKeySet
from mindtorch_v2._dispatch.registry import registry
from mindtorch_v2._dispatch.dispatcher import dispatch_with_keyset


def test_pipeline_key_precedes_functionalize():
    registry.register_schema("aten::pipe_order", "pipe_order(Tensor a) -> Tensor")

    def pipeline_impl(a):
        return "pipeline"

    def func_impl(a):
        return "functionalize"

    registry.register_kernel("aten::pipe_order", DispatchKey.Pipeline, pipeline_impl)
    registry.register_kernel("aten::pipe_order", DispatchKey.Functionalize, func_impl)

    keyset = DispatchKeySet({DispatchKey.Pipeline, DispatchKey.Functionalize})
    out = dispatch_with_keyset("pipe_order", keyset, None, 1)
    assert out == "pipeline"
