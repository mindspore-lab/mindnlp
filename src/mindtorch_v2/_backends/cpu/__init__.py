from ..common import convert as convert_backend
from ..common import view as view_backend
from ..meta import infer as meta_infer
from ..._dispatch.registry import registry
from .creation import (
    arange_create,
    empty_create,
    full_create,
    eye_create,
    linspace_create,
    logspace_create,
    ones_create,
    range_create,
    tensor_create,
    zeros_create,
)
from .ops import (
    add,
    matmul,
    mul,
    relu,
    sum_,
    all_,
    any_,
    argmax,
    argmin,
    count_nonzero,
    cumsum,
    cumprod,
    cummax,
    argsort,
    sort,
    topk,
    stack,
    cat,
    concatenate,
    hstack,
    vstack,
    row_stack,
    dstack,
    column_stack,
    pad_sequence,
    block_diag,
    tril,
    triu,
    diag,
    cartesian_prod,
    chunk,
    split,
    vsplit,
    hsplit,
    dsplit,
    unbind,
    allclose,
    isclose,
    equal,
    add_,
    mul_,
    relu_,
    zero_,
    contiguous,
    abs,
    neg,
    exp,
    log,
    sqrt,
    sin,
    cos,
    tan,
    tanh,
    sigmoid,
    floor,
    ceil,
    round,
    trunc,
    frac,
    pow,
    log2,
    log10,
    exp2,
    rsqrt,
    sign,
    signbit,
    isnan,
    isinf,
    isfinite,
    sinh,
    cosh,
    asinh,
    acosh,
    atanh,
    erf,
    erfc,
    softplus,
    clamp,
    clamp_min,
    clamp_max,
    relu6,
    hardtanh,
    min_,
    max_,
    amin,
    amax,
    fmin,
    fmax,
    where,
    atan,
    atan2,
    asin,
    acos,
    lerp,
    addcmul,
    addcdiv,
    logaddexp,
    logaddexp2,
    hypot,
    remainder,
    fmod,
    nonzero,
    masked_select,
    gather,
    scatter,
    take,
    take_along_dim,
    index_select,
    tril_indices,
    triu_indices,
    getitem,
    setitem,
)

registry.register("add", "cpu", add, meta=meta_infer.infer_binary)
registry.register("mul", "cpu", mul, meta=meta_infer.infer_binary)
registry.register("matmul", "cpu", matmul, meta=meta_infer.infer_matmul)
registry.register("relu", "cpu", relu, meta=meta_infer.infer_unary)
registry.register("abs", "cpu", abs, meta=meta_infer.infer_unary)
registry.register("neg", "cpu", neg, meta=meta_infer.infer_unary)
registry.register("exp", "cpu", exp, meta=meta_infer.infer_unary)
registry.register("log", "cpu", log, meta=meta_infer.infer_unary)
registry.register("sqrt", "cpu", sqrt, meta=meta_infer.infer_unary)
registry.register("sin", "cpu", sin, meta=meta_infer.infer_unary)
registry.register("cos", "cpu", cos, meta=meta_infer.infer_unary)
registry.register("tan", "cpu", tan, meta=meta_infer.infer_unary)
registry.register("tanh", "cpu", tanh, meta=meta_infer.infer_unary)
registry.register("sigmoid", "cpu", sigmoid, meta=meta_infer.infer_unary)
registry.register("floor", "cpu", floor, meta=meta_infer.infer_unary)
registry.register("ceil", "cpu", ceil, meta=meta_infer.infer_unary)
registry.register("round", "cpu", round, meta=meta_infer.infer_unary)
registry.register("trunc", "cpu", trunc, meta=meta_infer.infer_unary)
registry.register("frac", "cpu", frac, meta=meta_infer.infer_unary)
registry.register("pow", "cpu", pow, meta=meta_infer.infer_binary)
registry.register("log2", "cpu", log2, meta=meta_infer.infer_unary)
registry.register("log10", "cpu", log10, meta=meta_infer.infer_unary)
registry.register("exp2", "cpu", exp2, meta=meta_infer.infer_unary)
registry.register("rsqrt", "cpu", rsqrt, meta=meta_infer.infer_unary)
registry.register("sign", "cpu", sign, meta=meta_infer.infer_unary)
registry.register("signbit", "cpu", signbit, meta=meta_infer.infer_unary_bool)
registry.register("isnan", "cpu", isnan, meta=meta_infer.infer_unary_bool)
registry.register("isinf", "cpu", isinf, meta=meta_infer.infer_unary_bool)
registry.register("isfinite", "cpu", isfinite, meta=meta_infer.infer_unary_bool)
registry.register("sinh", "cpu", sinh, meta=meta_infer.infer_unary)
registry.register("cosh", "cpu", cosh, meta=meta_infer.infer_unary)
registry.register("asinh", "cpu", asinh, meta=meta_infer.infer_unary)
registry.register("acosh", "cpu", acosh, meta=meta_infer.infer_unary)
registry.register("atanh", "cpu", atanh, meta=meta_infer.infer_unary)
registry.register("erf", "cpu", erf, meta=meta_infer.infer_unary)
registry.register("erfc", "cpu", erfc, meta=meta_infer.infer_unary)
registry.register("softplus", "cpu", softplus, meta=meta_infer.infer_unary)
registry.register("clamp", "cpu", clamp, meta=meta_infer.infer_unary)
registry.register("clamp_min", "cpu", clamp_min, meta=meta_infer.infer_unary)
registry.register("clamp_max", "cpu", clamp_max, meta=meta_infer.infer_unary)
registry.register("relu6", "cpu", relu6, meta=meta_infer.infer_unary)
registry.register("hardtanh", "cpu", hardtanh, meta=meta_infer.infer_unary)
registry.register("min", "cpu", min_, meta=meta_infer.infer_binary)
registry.register("max", "cpu", max_, meta=meta_infer.infer_binary)
registry.register("amin", "cpu", amin, meta=meta_infer.infer_sum)
registry.register("amax", "cpu", amax, meta=meta_infer.infer_sum)
registry.register("fmin", "cpu", fmin, meta=meta_infer.infer_binary)
registry.register("fmax", "cpu", fmax, meta=meta_infer.infer_binary)
registry.register("where", "cpu", where, meta=meta_infer.infer_binary)
registry.register("atan", "cpu", atan, meta=meta_infer.infer_unary)
registry.register("atan2", "cpu", atan2, meta=meta_infer.infer_binary)
registry.register("asin", "cpu", asin, meta=meta_infer.infer_unary)
registry.register("acos", "cpu", acos, meta=meta_infer.infer_unary)
registry.register("lerp", "cpu", lerp, meta=meta_infer.infer_binary)
registry.register("addcmul", "cpu", addcmul, meta=meta_infer.infer_binary)
registry.register("addcdiv", "cpu", addcdiv, meta=meta_infer.infer_binary)
registry.register("logaddexp", "cpu", logaddexp, meta=meta_infer.infer_binary)
registry.register("logaddexp2", "cpu", logaddexp2, meta=meta_infer.infer_binary)
registry.register("hypot", "cpu", hypot, meta=meta_infer.infer_binary)
registry.register("remainder", "cpu", remainder, meta=meta_infer.infer_binary)
registry.register("fmod", "cpu", fmod, meta=meta_infer.infer_binary)
registry.register("nonzero", "cpu", nonzero, meta=meta_infer.infer_nonzero)
registry.register("masked_select", "cpu", masked_select, meta=meta_infer.infer_masked_select)
registry.register("gather", "cpu", gather, meta=meta_infer.infer_gather)
registry.register("scatter", "cpu", scatter, meta=meta_infer.infer_scatter)
registry.register("take", "cpu", take, meta=meta_infer.infer_take)
registry.register("take_along_dim", "cpu", take_along_dim, meta=meta_infer.infer_take_along_dim)
registry.register("index_select", "cpu", index_select, meta=meta_infer.infer_index_select)
registry.register("tril_indices", "cpu", tril_indices, meta=meta_infer.infer_tril_indices)
registry.register("triu_indices", "cpu", triu_indices, meta=meta_infer.infer_triu_indices)
registry.register("getitem", "cpu", getitem)
registry.register("setitem", "cpu", setitem)
registry.register("contiguous", "cpu", contiguous, meta=meta_infer.infer_unary)
registry.register("sum", "cpu", sum_, meta=meta_infer.infer_sum)
registry.register("all", "cpu", all_, meta=meta_infer.infer_reduce_bool)
registry.register("any", "cpu", any_, meta=meta_infer.infer_reduce_bool)
registry.register("argmax", "cpu", argmax, meta=meta_infer.infer_argmax)
registry.register("argmin", "cpu", argmin, meta=meta_infer.infer_argmax)
registry.register("count_nonzero", "cpu", count_nonzero, meta=meta_infer.infer_argmax)
registry.register("cumsum", "cpu", cumsum, meta=meta_infer.infer_unary)
registry.register("cumprod", "cpu", cumprod, meta=meta_infer.infer_unary)
registry.register("cummax", "cpu", cummax, meta=meta_infer.infer_cummax)
registry.register("argsort", "cpu", argsort, meta=meta_infer.infer_argsort)
registry.register("sort", "cpu", sort, meta=meta_infer.infer_sort)
registry.register("topk", "cpu", topk, meta=meta_infer.infer_topk)
registry.register("stack", "cpu", stack, meta=meta_infer.infer_stack)
registry.register("cat", "cpu", cat, meta=meta_infer.infer_cat)
registry.register("concat", "cpu", cat, meta=meta_infer.infer_cat)
registry.register("concatenate", "cpu", concatenate, meta=meta_infer.infer_cat)
registry.register("hstack", "cpu", hstack, meta=meta_infer.infer_hstack)
registry.register("vstack", "cpu", vstack, meta=meta_infer.infer_vstack)
registry.register("row_stack", "cpu", row_stack, meta=meta_infer.infer_vstack)
registry.register("dstack", "cpu", dstack, meta=meta_infer.infer_dstack)
registry.register("column_stack", "cpu", column_stack, meta=meta_infer.infer_column_stack)
registry.register("pad_sequence", "cpu", pad_sequence, meta=meta_infer.infer_pad_sequence)
registry.register("block_diag", "cpu", block_diag, meta=meta_infer.infer_block_diag)
registry.register("tril", "cpu", tril, meta=meta_infer.infer_unary)
registry.register("triu", "cpu", triu, meta=meta_infer.infer_unary)
registry.register("diag", "cpu", diag, meta=meta_infer.infer_diag)
registry.register("cartesian_prod", "cpu", cartesian_prod, meta=meta_infer.infer_cartesian_prod)
registry.register("chunk", "cpu", chunk, meta=meta_infer.infer_chunk)
registry.register("split", "cpu", split, meta=meta_infer.infer_split)
registry.register("vsplit", "cpu", vsplit, meta=meta_infer.infer_vsplit)
registry.register("hsplit", "cpu", hsplit, meta=meta_infer.infer_hsplit)
registry.register("dsplit", "cpu", dsplit, meta=meta_infer.infer_dsplit)
registry.register("unbind", "cpu", unbind, meta=meta_infer.infer_unbind)
registry.register("allclose", "cpu", allclose, meta=meta_infer.infer_reduce_bool)
registry.register("isclose", "cpu", isclose, meta=meta_infer.infer_unary_bool)
registry.register("equal", "cpu", equal, meta=meta_infer.infer_reduce_bool)
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
registry.register("arange", "cpu", arange_create)
registry.register("linspace", "cpu", linspace_create)
registry.register("full", "cpu", full_create)
registry.register("logspace", "cpu", logspace_create)
registry.register("eye", "cpu", eye_create)
registry.register("range", "cpu", range_create)
