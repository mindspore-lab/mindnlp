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
    cosh,
    exp,
    exp2,
    erf,
    erfc,
    floor,
    frac,
    hardtanh,
    isfinite,
    isinf,
    isnan,
    log,
    log10,
    log2,
    matmul,
    mul,
    neg,
    relu6,
    sign,
    signbit,
    pow,
    relu,
    round,
    rsqrt,
    sigmoid,
    sin,
    sinh,
    softplus,
    sqrt,
    sum_,
    tan,
    tanh,
    trunc,
    clamp,
    clamp_min,
    clamp_max,
    amin,
    amax,
    argmax,
    argmin,
    add_,
    mul_,
    relu_,
    zero_,
    contiguous,
    getitem,
    setitem,
    all_,
    any_,
    count_nonzero,
)
from .runtime import is_available, _model_dir, _probe_model_dirs
from . import allocator

registry.register("add", "npu", add, meta=meta_infer.infer_binary)
registry.register("mul", "npu", mul, meta=meta_infer.infer_binary)
registry.register("matmul", "npu", matmul, meta=meta_infer.infer_matmul)
registry.register("relu", "npu", relu, meta=meta_infer.infer_unary)
registry.register("contiguous", "npu", contiguous, meta=meta_infer.infer_unary)
registry.register("sum", "npu", sum_, meta=meta_infer.infer_sum)

registry.register("all", "npu", all_, meta=meta_infer.infer_reduce_bool)
registry.register("any", "npu", any_, meta=meta_infer.infer_reduce_bool)
registry.register("count_nonzero", "npu", count_nonzero, meta=meta_infer.infer_argmax)
registry.register("abs", "npu", abs, meta=meta_infer.infer_unary)
registry.register("neg", "npu", neg, meta=meta_infer.infer_unary)
registry.register("sign", "npu", sign, meta=meta_infer.infer_unary)
registry.register("signbit", "npu", signbit, meta=meta_infer.infer_unary_bool)
registry.register("isfinite", "npu", isfinite, meta=meta_infer.infer_unary_bool)
registry.register("isinf", "npu", isinf, meta=meta_infer.infer_unary_bool)
registry.register("isnan", "npu", isnan, meta=meta_infer.infer_unary_bool)
registry.register("exp", "npu", exp, meta=meta_infer.infer_unary)
registry.register("log", "npu", log, meta=meta_infer.infer_unary)
registry.register("sqrt", "npu", sqrt, meta=meta_infer.infer_unary)
registry.register("rsqrt", "npu", rsqrt, meta=meta_infer.infer_unary)
registry.register("sin", "npu", sin, meta=meta_infer.infer_unary)
registry.register("cos", "npu", cos, meta=meta_infer.infer_unary)
registry.register("tan", "npu", tan, meta=meta_infer.infer_unary)
registry.register("tanh", "npu", tanh, meta=meta_infer.infer_unary)
registry.register("sigmoid", "npu", sigmoid, meta=meta_infer.infer_unary)
registry.register("sinh", "npu", sinh, meta=meta_infer.infer_unary)
registry.register("cosh", "npu", cosh, meta=meta_infer.infer_unary)
registry.register("erf", "npu", erf, meta=meta_infer.infer_unary)
registry.register("erfc", "npu", erfc, meta=meta_infer.infer_unary)
registry.register("floor", "npu", floor, meta=meta_infer.infer_unary)
registry.register("ceil", "npu", ceil, meta=meta_infer.infer_unary)
registry.register("round", "npu", round, meta=meta_infer.infer_unary)
registry.register("trunc", "npu", trunc, meta=meta_infer.infer_unary)
registry.register("frac", "npu", frac, meta=meta_infer.infer_unary)
registry.register("log2", "npu", log2, meta=meta_infer.infer_unary)
registry.register("log10", "npu", log10, meta=meta_infer.infer_unary)
registry.register("exp2", "npu", exp2, meta=meta_infer.infer_unary)
registry.register("softplus", "npu", softplus, meta=meta_infer.infer_unary)
registry.register("clamp", "npu", clamp, meta=meta_infer.infer_unary)
registry.register("clamp_min", "npu", clamp_min, meta=meta_infer.infer_unary)
registry.register("clamp_max", "npu", clamp_max, meta=meta_infer.infer_unary)
registry.register("relu6", "npu", relu6, meta=meta_infer.infer_unary)
registry.register("hardtanh", "npu", hardtanh, meta=meta_infer.infer_unary)
registry.register("pow", "npu", pow, meta=meta_infer.infer_binary)

registry.register("amin", "npu", amin, meta=meta_infer.infer_sum)
registry.register("amax", "npu", amax, meta=meta_infer.infer_sum)
registry.register("argmax", "npu", argmax, meta=meta_infer.infer_argmax)
registry.register("argmin", "npu", argmin, meta=meta_infer.infer_argmax)
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
registry.register("getitem", "npu", getitem)
registry.register("setitem", "npu", setitem)

__all__ = ["is_available", "_probe_model_dirs", "_model_dir", "allocator"]
