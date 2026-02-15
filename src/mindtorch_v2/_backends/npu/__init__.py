from ..common import convert as convert_backend
from ..common import view as view_backend
from ..meta import infer as meta_infer
from ..._dispatch.registry import registry
from .creation import empty_create, ones_create, tensor_create, zeros_create
from .ops import (
    abs,
    add,
    ceil,
    cos,
    exp,
    exp2,
    floor,
    frac,
    log,
    log10,
    log2,
    matmul,
    mul,
    neg,
    pow,
    relu,
    round,
    rsqrt,
    sigmoid,
    sin,
    sqrt,
    sum_,
    tan,
    tanh,
    trunc,
    add_,
    mul_,
    relu_,
    zero_,
    contiguous,
)
from .runtime import is_available, _model_dir, _probe_model_dirs
from . import allocator

registry.register("add", "npu", add, meta=meta_infer.infer_binary)
registry.register("mul", "npu", mul, meta=meta_infer.infer_binary)
registry.register("matmul", "npu", matmul, meta=meta_infer.infer_matmul)
registry.register("relu", "npu", relu, meta=meta_infer.infer_unary)
registry.register("contiguous", "npu", contiguous, meta=meta_infer.infer_unary)
registry.register("sum", "npu", sum_, meta=meta_infer.infer_sum)
registry.register("abs", "npu", abs, meta=meta_infer.infer_unary)
registry.register("neg", "npu", neg, meta=meta_infer.infer_unary)
registry.register("exp", "npu", exp, meta=meta_infer.infer_unary)
registry.register("log", "npu", log, meta=meta_infer.infer_unary)
registry.register("sqrt", "npu", sqrt, meta=meta_infer.infer_unary)
registry.register("rsqrt", "npu", rsqrt, meta=meta_infer.infer_unary)
registry.register("sin", "npu", sin, meta=meta_infer.infer_unary)
registry.register("cos", "npu", cos, meta=meta_infer.infer_unary)
registry.register("tan", "npu", tan, meta=meta_infer.infer_unary)
registry.register("tanh", "npu", tanh, meta=meta_infer.infer_unary)
registry.register("sigmoid", "npu", sigmoid, meta=meta_infer.infer_unary)
registry.register("floor", "npu", floor, meta=meta_infer.infer_unary)
registry.register("ceil", "npu", ceil, meta=meta_infer.infer_unary)
registry.register("round", "npu", round, meta=meta_infer.infer_unary)
registry.register("trunc", "npu", trunc, meta=meta_infer.infer_unary)
registry.register("frac", "npu", frac, meta=meta_infer.infer_unary)
registry.register("log2", "npu", log2, meta=meta_infer.infer_unary)
registry.register("log10", "npu", log10, meta=meta_infer.infer_unary)
registry.register("exp2", "npu", exp2, meta=meta_infer.infer_unary)
registry.register("pow", "npu", pow, meta=meta_infer.infer_binary)
registry.register("add_", "npu", add_, meta=meta_infer.infer_binary)
registry.register("mul_", "npu", mul_, meta=meta_infer.infer_binary)
registry.register("relu_", "npu", relu_, meta=meta_infer.infer_unary)
registry.register("zero_", "npu", zero_, meta=meta_infer.infer_unary)
registry.register("reshape", "npu", view_backend.reshape, meta=meta_infer.infer_view)
registry.register("view", "npu", view_backend.view, meta=meta_infer.infer_view)
registry.register("transpose", "npu", view_backend.transpose, meta=meta_infer.infer_transpose)
registry.register("to", "npu", convert_backend.to_device)

registry.register("tensor", "npu", tensor_create)
registry.register("zeros", "npu", zeros_create)
registry.register("ones", "npu", ones_create)
registry.register("empty", "npu", empty_create)

__all__ = ["is_available", "_probe_model_dirs", "_model_dir", "allocator"]
