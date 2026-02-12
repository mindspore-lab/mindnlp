from ..common import convert as convert_backend
from ..common import view as view_backend
from ..._dispatch.registry import registry
from .creation import (
    empty_create,
    empty_create_meta,
    ones_create,
    ones_create_meta,
    tensor_create,
    tensor_create_meta,
    zeros_create,
    zeros_create_meta,
)
from .meta import (
    _meta_binary,
    _meta_sum,
    _meta_transpose,
    _meta_unary,
    _meta_view,
)
from .ops import add, matmul, mul, relu, sum_

registry.register("add", "cpu", add, meta=_meta_binary)
registry.register("mul", "cpu", mul, meta=_meta_binary)
registry.register("matmul", "cpu", matmul)
registry.register("relu", "cpu", relu, meta=_meta_unary)
registry.register("sum", "cpu", sum_, meta=_meta_sum)
registry.register("reshape", "cpu", view_backend.reshape, meta=_meta_view)
registry.register("view", "cpu", view_backend.view, meta=_meta_view)
registry.register("transpose", "cpu", view_backend.transpose, meta=_meta_transpose)
registry.register("to", "cpu", convert_backend.to_device)

registry.register("tensor", "cpu", tensor_create)
registry.register("zeros", "cpu", zeros_create)
registry.register("ones", "cpu", ones_create)
registry.register("empty", "cpu", empty_create)
