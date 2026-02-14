from ..common import convert as convert_backend
from ..common import view as view_backend
from ..meta import infer as meta_infer
from ..._dispatch.registry import registry
from .creation import empty_create, ones_create, tensor_create, zeros_create
from .ops import add, matmul, mul, relu, sum_, add_, mul_, relu_, zero_

registry.register("add", "cpu", add, meta=meta_infer.infer_binary)
registry.register("mul", "cpu", mul, meta=meta_infer.infer_binary)
registry.register("matmul", "cpu", matmul, meta=meta_infer.infer_matmul)
registry.register("relu", "cpu", relu, meta=meta_infer.infer_unary)
registry.register("sum", "cpu", sum_, meta=meta_infer.infer_sum)
registry.register("add_", "cpu", add_, meta=meta_infer.infer_binary)
registry.register("mul_", "cpu", mul_, meta=meta_infer.infer_binary)
registry.register("relu_", "cpu", relu_, meta=meta_infer.infer_unary)
registry.register("zero_", "cpu", zero_, meta=meta_infer.infer_unary)
registry.register("reshape", "cpu", view_backend.reshape, meta=meta_infer.infer_view)
registry.register("view", "cpu", view_backend.view, meta=meta_infer.infer_view)
registry.register("transpose", "cpu", view_backend.transpose, meta=meta_infer.infer_transpose)
registry.register("to", "cpu", convert_backend.to_device)

registry.register("tensor", "cpu", tensor_create)
registry.register("zeros", "cpu", zeros_create)
registry.register("ones", "cpu", ones_create)
registry.register("empty", "cpu", empty_create)
