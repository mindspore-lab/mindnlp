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
    stack,
    cat,
    concatenate,
    chunk,
    split,
    vsplit,
    hsplit,
    dsplit,
    unbind,
    hstack,
    vstack,
    row_stack,
    dstack,
    column_stack,
    where,
    sub,
    div,
    acosh,
    addcdiv,
    addcmul,
    allclose,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    equal,
    fmax,
    fmin,
    fmod,
    hypot,
    isclose,
    lerp,
    logaddexp,
    logaddexp2,
    max_,
    min_,
    remainder,
    where,
    acos,
    mean,
    softmax,
    log_softmax,
    gelu,
    layer_norm,
    embedding,
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
registry.register("min", "npu", min_, meta=meta_infer.infer_binary)
registry.register("max", "npu", max_, meta=meta_infer.infer_binary)
registry.register("pow", "npu", pow, meta=meta_infer.infer_binary)

registry.register("amin", "npu", amin, meta=meta_infer.infer_sum)
registry.register("amax", "npu", amax, meta=meta_infer.infer_sum)
registry.register("argmax", "npu", argmax, meta=meta_infer.infer_argmax)
registry.register("argmin", "npu", argmin, meta=meta_infer.infer_argmax)
registry.register("add_", "npu", add_, meta=meta_infer.infer_binary)
registry.register("mul_", "npu", mul_, meta=meta_infer.infer_binary)
registry.register("relu_", "npu", relu_, meta=meta_infer.infer_unary)
registry.register("zero_", "npu", zero_, meta=meta_infer.infer_unary)
registry.register("sub", "npu", sub, meta=meta_infer.infer_binary)
registry.register("div", "npu", div, meta=meta_infer.infer_binary)
registry.register("true_divide", "npu", div, meta=meta_infer.infer_binary)
registry.register("asin", "npu", asin, meta=meta_infer.infer_unary)
registry.register("acos", "npu", acos, meta=meta_infer.infer_unary)
registry.register("atan", "npu", atan, meta=meta_infer.infer_unary)
registry.register("atan2", "npu", atan2, meta=meta_infer.infer_binary)
registry.register("asinh", "npu", asinh, meta=meta_infer.infer_unary)
registry.register("acosh", "npu", acosh, meta=meta_infer.infer_unary)
registry.register("atanh", "npu", atanh, meta=meta_infer.infer_unary)
registry.register("min_", "npu", min_, meta=meta_infer.infer_binary)
registry.register("max_", "npu", max_, meta=meta_infer.infer_binary)
registry.register("fmin", "npu", fmin, meta=meta_infer.infer_binary)
registry.register("fmax", "npu", fmax, meta=meta_infer.infer_binary)
registry.register("where", "npu", where, meta=meta_infer.infer_binary)
registry.register("lerp", "npu", lerp, meta=meta_infer.infer_binary)
registry.register("addcmul", "npu", addcmul, meta=meta_infer.infer_binary)
registry.register("addcdiv", "npu", addcdiv, meta=meta_infer.infer_binary)
registry.register("logaddexp", "npu", logaddexp, meta=meta_infer.infer_binary)
registry.register("logaddexp2", "npu", logaddexp2, meta=meta_infer.infer_binary)
registry.register("hypot", "npu", hypot, meta=meta_infer.infer_binary)
registry.register("remainder", "npu", remainder, meta=meta_infer.infer_binary)
registry.register("fmod", "npu", fmod, meta=meta_infer.infer_binary)
registry.register("allclose", "npu", allclose, meta=meta_infer.infer_reduce_bool)
registry.register("isclose", "npu", isclose, meta=meta_infer.infer_unary_bool)
registry.register("equal", "npu", equal, meta=meta_infer.infer_reduce_bool)
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

registry.register("stack", "npu", stack, meta=meta_infer.infer_stack)
registry.register("cat", "npu", cat, meta=meta_infer.infer_cat)
registry.register("concat", "npu", cat, meta=meta_infer.infer_cat)
registry.register("concatenate", "npu", concatenate, meta=meta_infer.infer_cat)
registry.register("chunk", "npu", chunk)
registry.register("split", "npu", split)
registry.register("vsplit", "npu", vsplit)
registry.register("hsplit", "npu", hsplit)
registry.register("dsplit", "npu", dsplit)
registry.register("unbind", "npu", unbind)
registry.register("hstack", "npu", hstack, meta=meta_infer.infer_hstack)
registry.register("vstack", "npu", vstack, meta=meta_infer.infer_vstack)
registry.register("row_stack", "npu", row_stack, meta=meta_infer.infer_vstack)
registry.register("dstack", "npu", dstack, meta=meta_infer.infer_dstack)
registry.register("column_stack", "npu", column_stack, meta=meta_infer.infer_column_stack)
registry.register("where", "npu", where, meta=meta_infer.infer_binary)

# Critical tier operations
registry.register("mean", "npu", mean, meta=meta_infer.infer_sum)
registry.register("softmax", "npu", softmax, meta=meta_infer.infer_unary)
registry.register("log_softmax", "npu", log_softmax, meta=meta_infer.infer_unary)
registry.register("gelu", "npu", gelu, meta=meta_infer.infer_unary)
registry.register("layer_norm", "npu", layer_norm, meta=meta_infer.infer_unary)
registry.register("embedding", "npu", embedding, meta=meta_infer.infer_binary)

__all__ = ["is_available", "_probe_model_dirs", "_model_dir", "allocator"]
