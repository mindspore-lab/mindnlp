from ..common import convert as convert_backend
from ..common import view as view_backend
from ..meta import infer as meta_infer
from ..._dispatch.registry import registry
from .creation import (
    arange_create,
    empty_create,
    eye_create,
    full_create,
    linspace_create,
    logspace_create,
    ones_create,
    range_create,
    rand_create,
    randn_create,
    randint_create,
    randperm_create,
    tensor_create,
    zeros_create,
)
from .ops import (
    add,
    matmul,
    mm,
    bmm,
    mul,
    div,
    true_divide,
    relu,
    sum_,
    mean_,
    std_,
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
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    add_,
    mul_,
    relu_,
    zero_,
    uniform_,
    normal_,
    bernoulli_,
    exponential_,
    log_normal_,
    cauchy_,
    geometric_,
    fill_,
    clamp_,
    copy_,
    erfinv_,
    sub_,
    div_,
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
    square,
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
    gelu,
    silu,
    leaky_relu,
    elu,
    mish,
    prelu,
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
    flip,
    roll,
    rot90,
    repeat,
    repeat_interleave,
    tile,
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
    batch_norm,
    instance_norm,
    group_norm,
    layer_norm,
    dropout,
    pad,
    softmax,
    log_softmax,
    linalg_qr,
    narrow,
    select,
    expand,
    masked_fill,
    masked_fill_,
    index_put_,
    index_put,
    index_copy_,
    index_fill_,
    index_add_,
    scatter_,
    scatter_add_,
    masked_scatter_,
    unfold,
    embedding,
    var_,
    norm_,
    prod_,
    floor_divide,
    rms_norm,
    conv2d,
    conv1d,
    conv_transpose2d,
    conv_transpose1d,
    max_pool2d,
    avg_pool2d,
    adaptive_avg_pool2d,
    logical_and,
    logical_or,
    logical_not,
    # New math ops
    sub,
    log1p,
    expm1,
    reciprocal,
    maximum,
    minimum,
    dot,
    outer,
    inner,
    mv,
    cross,
    tensordot,
    einsum,
    # New logical ops
    logical_xor,
    # New bitwise ops
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_not,
    # New random in-place op
    randint_,
    random_,
    # New shape ops
    flatten,
    unflatten,
    broadcast_to,
    movedim,
    diagonal,
    # New search ops
    unique,
    searchsorted,
    kthvalue,
    median,
    # New GROUP C ops
    logsumexp,
    trace,
    det,
    matrix_power,
    dist,
    renorm,
    nansum,
    nanmean,
    argwhere,
    baddbmm,
    cummin,
)

# ---------------------------------------------------------------------------
# Core arithmetic / matmul
# ---------------------------------------------------------------------------
registry.register("add", "mps", add, meta=meta_infer.infer_binary)
registry.register("mul", "mps", mul, meta=meta_infer.infer_binary)
registry.register("div", "mps", div, meta=meta_infer.infer_binary)
registry.register("true_divide", "mps", true_divide, meta=meta_infer.infer_binary)
registry.register("matmul", "mps", matmul, meta=meta_infer.infer_matmul)
registry.register("mm", "mps", mm, meta=meta_infer.infer_matmul)
registry.register("bmm", "mps", bmm, meta=meta_infer.infer_matmul)

# ---------------------------------------------------------------------------
# Unary math / activations
# ---------------------------------------------------------------------------
registry.register("relu", "mps", relu, meta=meta_infer.infer_unary)
registry.register("abs", "mps", abs, meta=meta_infer.infer_unary)
registry.register("neg", "mps", neg, meta=meta_infer.infer_unary)
registry.register("exp", "mps", exp, meta=meta_infer.infer_unary)
registry.register("log", "mps", log, meta=meta_infer.infer_unary)
registry.register("sqrt", "mps", sqrt, meta=meta_infer.infer_unary)
registry.register("sin", "mps", sin, meta=meta_infer.infer_unary)
registry.register("cos", "mps", cos, meta=meta_infer.infer_unary)
registry.register("tan", "mps", tan, meta=meta_infer.infer_unary)
registry.register("tanh", "mps", tanh, meta=meta_infer.infer_unary)
registry.register("sigmoid", "mps", sigmoid, meta=meta_infer.infer_unary)
registry.register("floor", "mps", floor, meta=meta_infer.infer_unary)
registry.register("ceil", "mps", ceil, meta=meta_infer.infer_unary)
registry.register("round", "mps", round, meta=meta_infer.infer_unary)
registry.register("trunc", "mps", trunc, meta=meta_infer.infer_unary)
registry.register("frac", "mps", frac, meta=meta_infer.infer_unary)
registry.register("pow", "mps", pow, meta=meta_infer.infer_binary)
registry.register("log2", "mps", log2, meta=meta_infer.infer_unary)
registry.register("log10", "mps", log10, meta=meta_infer.infer_unary)
registry.register("exp2", "mps", exp2, meta=meta_infer.infer_unary)
registry.register("rsqrt", "mps", rsqrt, meta=meta_infer.infer_unary)
registry.register("sign", "mps", sign, meta=meta_infer.infer_unary)
registry.register("square", "mps", square, meta=meta_infer.infer_unary)
registry.register("signbit", "mps", signbit, meta=meta_infer.infer_unary_bool)
registry.register("isnan", "mps", isnan, meta=meta_infer.infer_unary_bool)
registry.register("isinf", "mps", isinf, meta=meta_infer.infer_unary_bool)
registry.register("isfinite", "mps", isfinite, meta=meta_infer.infer_unary_bool)
registry.register("sinh", "mps", sinh, meta=meta_infer.infer_unary)
registry.register("cosh", "mps", cosh, meta=meta_infer.infer_unary)
registry.register("asinh", "mps", asinh, meta=meta_infer.infer_unary)
registry.register("acosh", "mps", acosh, meta=meta_infer.infer_unary)
registry.register("atanh", "mps", atanh, meta=meta_infer.infer_unary)
registry.register("erf", "mps", erf, meta=meta_infer.infer_unary)
registry.register("erfc", "mps", erfc, meta=meta_infer.infer_unary)
registry.register("softplus", "mps", softplus, meta=meta_infer.infer_unary)
registry.register("clamp", "mps", clamp, meta=meta_infer.infer_unary)
registry.register("clamp_min", "mps", clamp_min, meta=meta_infer.infer_unary)
registry.register("clamp_max", "mps", clamp_max, meta=meta_infer.infer_unary)
registry.register("relu6", "mps", relu6, meta=meta_infer.infer_unary)
registry.register("hardtanh", "mps", hardtanh, meta=meta_infer.infer_unary)
registry.register("silu", "mps", silu, meta=meta_infer.infer_unary)
registry.register("leaky_relu", "mps", leaky_relu, meta=meta_infer.infer_unary)
registry.register("elu", "mps", elu, meta=meta_infer.infer_unary)
registry.register("mish", "mps", mish, meta=meta_infer.infer_unary)
registry.register("prelu", "mps", prelu, meta=meta_infer.infer_binary)

# ---------------------------------------------------------------------------
# Comparison / min / max
# ---------------------------------------------------------------------------
registry.register("min", "mps", min_, meta=meta_infer.infer_binary)
registry.register("max", "mps", max_, meta=meta_infer.infer_binary)
registry.register("amin", "mps", amin, meta=meta_infer.infer_sum)
registry.register("amax", "mps", amax, meta=meta_infer.infer_sum)
registry.register("fmin", "mps", fmin, meta=meta_infer.infer_binary)
registry.register("fmax", "mps", fmax, meta=meta_infer.infer_binary)
registry.register("where", "mps", where, meta=meta_infer.infer_binary)
registry.register("atan", "mps", atan, meta=meta_infer.infer_unary)
registry.register("atan2", "mps", atan2, meta=meta_infer.infer_binary)
registry.register("asin", "mps", asin, meta=meta_infer.infer_unary)
registry.register("acos", "mps", acos, meta=meta_infer.infer_unary)
registry.register("lerp", "mps", lerp, meta=meta_infer.infer_binary)
registry.register("addcmul", "mps", addcmul, meta=meta_infer.infer_binary)
registry.register("addcdiv", "mps", addcdiv, meta=meta_infer.infer_binary)
registry.register("logaddexp", "mps", logaddexp, meta=meta_infer.infer_binary)
registry.register("logaddexp2", "mps", logaddexp2, meta=meta_infer.infer_binary)
registry.register("hypot", "mps", hypot, meta=meta_infer.infer_binary)
registry.register("remainder", "mps", remainder, meta=meta_infer.infer_binary)
registry.register("fmod", "mps", fmod, meta=meta_infer.infer_binary)

# ---------------------------------------------------------------------------
# Shape / manipulation
# ---------------------------------------------------------------------------
registry.register("flip", "mps", flip, meta=meta_infer.infer_flip)
registry.register("roll", "mps", roll, meta=meta_infer.infer_roll)
registry.register("rot90", "mps", rot90, meta=meta_infer.infer_rot90)
registry.register("repeat", "mps", repeat, meta=meta_infer.infer_repeat)
registry.register("repeat_interleave", "mps", repeat_interleave, meta=meta_infer.infer_repeat_interleave)
registry.register("tile", "mps", tile, meta=meta_infer.infer_tile)
registry.register("nonzero", "mps", nonzero, meta=meta_infer.infer_nonzero)
registry.register("masked_select", "mps", masked_select, meta=meta_infer.infer_masked_select)
registry.register("gather", "mps", gather, meta=meta_infer.infer_gather)
registry.register("scatter", "mps", scatter, meta=meta_infer.infer_scatter)
registry.register("take", "mps", take, meta=meta_infer.infer_take)
registry.register("take_along_dim", "mps", take_along_dim, meta=meta_infer.infer_take_along_dim)
registry.register("index_select", "mps", index_select, meta=meta_infer.infer_index_select)
registry.register("tril_indices", "mps", tril_indices, meta=meta_infer.infer_tril_indices)
registry.register("triu_indices", "mps", triu_indices, meta=meta_infer.infer_triu_indices)
registry.register("getitem", "mps", getitem)
registry.register("setitem", "mps", setitem)
registry.register("contiguous", "mps", contiguous, meta=meta_infer.infer_unary)

# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
registry.register("sum", "mps", sum_, meta=meta_infer.infer_sum)
registry.register("mean", "mps", mean_, meta=meta_infer.infer_sum)
registry.register("std", "mps", std_, meta=meta_infer.infer_sum)
registry.register("all", "mps", all_, meta=meta_infer.infer_reduce_bool)
registry.register("any", "mps", any_, meta=meta_infer.infer_reduce_bool)
registry.register("argmax", "mps", argmax, meta=meta_infer.infer_argmax)
registry.register("argmin", "mps", argmin, meta=meta_infer.infer_argmax)
registry.register("count_nonzero", "mps", count_nonzero, meta=meta_infer.infer_argmax)
registry.register("cumsum", "mps", cumsum, meta=meta_infer.infer_unary)
registry.register("cumprod", "mps", cumprod, meta=meta_infer.infer_unary)
registry.register("cummax", "mps", cummax, meta=meta_infer.infer_cummax)
registry.register("argsort", "mps", argsort, meta=meta_infer.infer_argsort)
registry.register("sort", "mps", sort, meta=meta_infer.infer_sort)
registry.register("topk", "mps", topk, meta=meta_infer.infer_topk)

# ---------------------------------------------------------------------------
# Stack / cat / split
# ---------------------------------------------------------------------------
registry.register("stack", "mps", stack, meta=meta_infer.infer_stack)
registry.register("cat", "mps", cat, meta=meta_infer.infer_cat)
registry.register("concat", "mps", cat, meta=meta_infer.infer_cat)
registry.register("concatenate", "mps", concatenate, meta=meta_infer.infer_cat)
registry.register("hstack", "mps", hstack, meta=meta_infer.infer_hstack)
registry.register("vstack", "mps", vstack, meta=meta_infer.infer_vstack)
registry.register("row_stack", "mps", row_stack, meta=meta_infer.infer_vstack)
registry.register("dstack", "mps", dstack, meta=meta_infer.infer_dstack)
registry.register("column_stack", "mps", column_stack, meta=meta_infer.infer_column_stack)
registry.register("pad_sequence", "mps", pad_sequence, meta=meta_infer.infer_pad_sequence)
registry.register("block_diag", "mps", block_diag, meta=meta_infer.infer_block_diag)
registry.register("tril", "mps", tril, meta=meta_infer.infer_unary)
registry.register("triu", "mps", triu, meta=meta_infer.infer_unary)
registry.register("diag", "mps", diag, meta=meta_infer.infer_diag)
registry.register("cartesian_prod", "mps", cartesian_prod, meta=meta_infer.infer_cartesian_prod)
registry.register("chunk", "mps", chunk, meta=meta_infer.infer_chunk)
registry.register("split", "mps", split, meta=meta_infer.infer_split)
registry.register("vsplit", "mps", vsplit, meta=meta_infer.infer_vsplit)
registry.register("hsplit", "mps", hsplit, meta=meta_infer.infer_hsplit)
registry.register("dsplit", "mps", dsplit, meta=meta_infer.infer_dsplit)
registry.register("unbind", "mps", unbind, meta=meta_infer.infer_unbind)
registry.register("allclose", "mps", allclose, meta=meta_infer.infer_reduce_bool)
registry.register("isclose", "mps", isclose, meta=meta_infer.infer_unary_bool)
registry.register("equal", "mps", equal, meta=meta_infer.infer_reduce_bool)
registry.register("eq", "mps", eq, meta=meta_infer.infer_binary_bool)
registry.register("ne", "mps", ne, meta=meta_infer.infer_binary_bool)
registry.register("lt", "mps", lt, meta=meta_infer.infer_binary_bool)
registry.register("le", "mps", le, meta=meta_infer.infer_binary_bool)
registry.register("gt", "mps", gt, meta=meta_infer.infer_binary_bool)
registry.register("ge", "mps", ge, meta=meta_infer.infer_binary_bool)

# ---------------------------------------------------------------------------
# In-place ops
# ---------------------------------------------------------------------------
registry.register("add_", "mps", add_, meta=meta_infer.infer_binary)
registry.register("mul_", "mps", mul_, meta=meta_infer.infer_binary)
registry.register("relu_", "mps", relu_, meta=meta_infer.infer_unary)
registry.register("zero_", "mps", zero_, meta=meta_infer.infer_unary)
registry.register("uniform_", "mps", uniform_, meta=meta_infer.infer_unary)
registry.register("normal_", "mps", normal_, meta=meta_infer.infer_unary)
registry.register("bernoulli_", "mps", bernoulli_, meta=meta_infer.infer_unary)
registry.register("exponential_", "mps", exponential_, meta=meta_infer.infer_unary)
registry.register("log_normal_", "mps", log_normal_, meta=meta_infer.infer_unary)
registry.register("cauchy_", "mps", cauchy_, meta=meta_infer.infer_unary)
registry.register("geometric_", "mps", geometric_, meta=meta_infer.infer_unary)
registry.register("fill_", "mps", fill_, meta=meta_infer.infer_unary)
registry.register("clamp_", "mps", clamp_, meta=meta_infer.infer_unary)
registry.register("copy_", "mps", copy_, meta=meta_infer.infer_unary)
registry.register("erfinv_", "mps", erfinv_, meta=meta_infer.infer_unary)
registry.register("sub_", "mps", sub_, meta=meta_infer.infer_binary)
registry.register("div_", "mps", div_, meta=meta_infer.infer_binary)

# ---------------------------------------------------------------------------
# NN ops
# ---------------------------------------------------------------------------
registry.register("batch_norm", "mps", batch_norm, meta=meta_infer.infer_unary)
registry.register("instance_norm", "mps", instance_norm, meta=meta_infer.infer_unary)
registry.register("group_norm", "mps", group_norm, meta=meta_infer.infer_unary)
registry.register("layer_norm", "mps", layer_norm, meta=meta_infer.infer_unary)
registry.register("dropout", "mps", dropout, meta=meta_infer.infer_unary)
registry.register("gelu", "mps", gelu, meta=meta_infer.infer_unary)
registry.register("pad", "mps", pad, meta=meta_infer.infer_unary)
registry.register("softmax", "mps", softmax, meta=meta_infer.infer_unary)
registry.register("log_softmax", "mps", log_softmax, meta=meta_infer.infer_unary)
registry.register("embedding", "mps", embedding)

# ---------------------------------------------------------------------------
# View / reshape / permute
# ---------------------------------------------------------------------------
registry.register("reshape", "mps", view_backend.reshape, meta=meta_infer.infer_view)
registry.register("view", "mps", view_backend.view, meta=meta_infer.infer_view)
registry.register("transpose", "mps", view_backend.transpose, meta=meta_infer.infer_transpose)
registry.register("squeeze", "mps", view_backend.squeeze, meta=meta_infer.infer_view)
registry.register("unsqueeze", "mps", view_backend.unsqueeze, meta=meta_infer.infer_view)
registry.register("permute", "mps", view_backend.permute, meta=meta_infer.infer_view)
registry.register("to", "mps", convert_backend.to_device)

# ---------------------------------------------------------------------------
# Creation ops
# ---------------------------------------------------------------------------
registry.register("tensor", "mps", tensor_create)
registry.register("zeros", "mps", zeros_create)
registry.register("ones", "mps", ones_create)
registry.register("empty", "mps", empty_create)
registry.register("arange", "mps", arange_create)
registry.register("linspace", "mps", linspace_create)
registry.register("full", "mps", full_create)
registry.register("logspace", "mps", logspace_create)
registry.register("eye", "mps", eye_create)
registry.register("range", "mps", range_create)
registry.register("randn", "mps", randn_create)
registry.register("rand", "mps", rand_create)
registry.register("randint", "mps", randint_create)
registry.register("randperm", "mps", randperm_create)
registry.register("linalg_qr", "mps", linalg_qr)

# ---------------------------------------------------------------------------
# Indexing / selection
# ---------------------------------------------------------------------------
registry.register("narrow", "mps", narrow)
registry.register("select", "mps", select)
registry.register("expand", "mps", expand)
registry.register("masked_fill", "mps", masked_fill, meta=meta_infer.infer_unary)
registry.register("masked_fill_", "mps", masked_fill_, meta=meta_infer.infer_unary)
registry.register("index_put_", "mps", index_put_)
registry.register("index_put", "mps", index_put)
registry.register("index_copy_", "mps", index_copy_)
registry.register("index_fill_", "mps", index_fill_)
registry.register("index_add_", "mps", index_add_)
registry.register("scatter_", "mps", scatter_)
registry.register("scatter_add_", "mps", scatter_add_)
registry.register("masked_scatter_", "mps", masked_scatter_)
registry.register("unfold", "mps", unfold)

# ---------------------------------------------------------------------------
# Var / norm / prod / floor_divide / rms_norm
# ---------------------------------------------------------------------------
registry.register("var", "mps", var_, meta=meta_infer.infer_sum)
registry.register("norm", "mps", norm_, meta=meta_infer.infer_sum)
registry.register("prod", "mps", prod_, meta=meta_infer.infer_sum)
registry.register("floor_divide", "mps", floor_divide, meta=meta_infer.infer_binary)
registry.register("rms_norm", "mps", rms_norm, meta=meta_infer.infer_unary)

# ---------------------------------------------------------------------------
# Conv / pool
# ---------------------------------------------------------------------------
registry.register("conv2d", "mps", conv2d)
registry.register("conv1d", "mps", conv1d)
registry.register("conv_transpose2d", "mps", conv_transpose2d)
registry.register("conv_transpose1d", "mps", conv_transpose1d)
registry.register("max_pool2d", "mps", max_pool2d)
registry.register("avg_pool2d", "mps", avg_pool2d)
registry.register("adaptive_avg_pool2d", "mps", adaptive_avg_pool2d)

# ---------------------------------------------------------------------------
# Math ops
# ---------------------------------------------------------------------------
registry.register("sub", "mps", sub, meta=meta_infer.infer_binary)
registry.register("log1p", "mps", log1p, meta=meta_infer.infer_unary)
registry.register("expm1", "mps", expm1, meta=meta_infer.infer_unary)
registry.register("reciprocal", "mps", reciprocal, meta=meta_infer.infer_unary)
registry.register("maximum", "mps", maximum, meta=meta_infer.infer_binary)
registry.register("minimum", "mps", minimum, meta=meta_infer.infer_binary)
registry.register("dot", "mps", dot, meta=meta_infer.infer_dot)
registry.register("outer", "mps", outer, meta=meta_infer.infer_outer)
registry.register("inner", "mps", inner, meta=meta_infer.infer_binary)
registry.register("mv", "mps", mv, meta=meta_infer.infer_binary)
registry.register("cross", "mps", cross, meta=meta_infer.infer_binary)
registry.register("tensordot", "mps", tensordot)
registry.register("einsum", "mps", einsum)

# ---------------------------------------------------------------------------
# Logical ops
# ---------------------------------------------------------------------------
registry.register("logical_and", "mps", logical_and, meta=meta_infer.infer_binary_bool)
registry.register("logical_or", "mps", logical_or, meta=meta_infer.infer_binary_bool)
registry.register("logical_not", "mps", logical_not, meta=meta_infer.infer_unary_bool)
registry.register("logical_xor", "mps", logical_xor, meta=meta_infer.infer_binary_bool)

# ---------------------------------------------------------------------------
# Bitwise ops
# ---------------------------------------------------------------------------
registry.register("bitwise_and", "mps", bitwise_and, meta=meta_infer.infer_binary)
registry.register("bitwise_or", "mps", bitwise_or, meta=meta_infer.infer_binary)
registry.register("bitwise_xor", "mps", bitwise_xor, meta=meta_infer.infer_binary)
registry.register("bitwise_not", "mps", bitwise_not, meta=meta_infer.infer_unary)

# ---------------------------------------------------------------------------
# Random in-place
# ---------------------------------------------------------------------------
registry.register("randint_", "mps", randint_, meta=meta_infer.infer_unary)
registry.register("random_", "mps", random_, meta=meta_infer.infer_unary)

# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------
registry.register("flatten", "mps", flatten, meta=meta_infer.infer_flatten)
registry.register("unflatten", "mps", unflatten, meta=meta_infer.infer_unflatten)
registry.register("broadcast_to", "mps", broadcast_to, meta=meta_infer.infer_broadcast_to)
registry.register("movedim", "mps", movedim, meta=meta_infer.infer_movedim)
registry.register("diagonal", "mps", diagonal, meta=meta_infer.infer_diagonal)

# ---------------------------------------------------------------------------
# Search / sort ops
# ---------------------------------------------------------------------------
registry.register("unique", "mps", unique)
registry.register("searchsorted", "mps", searchsorted)
registry.register("kthvalue", "mps", kthvalue)
registry.register("median", "mps", median)

# ---------------------------------------------------------------------------
# GROUP C ops
# ---------------------------------------------------------------------------
registry.register("logsumexp", "mps", logsumexp, meta=meta_infer.infer_sum)
registry.register("trace", "mps", trace)
registry.register("det", "mps", det)
registry.register("matrix_power", "mps", matrix_power, meta=meta_infer.infer_unary)
registry.register("dist", "mps", dist)
registry.register("renorm", "mps", renorm, meta=meta_infer.infer_unary)
registry.register("nansum", "mps", nansum, meta=meta_infer.infer_sum)
registry.register("nanmean", "mps", nanmean, meta=meta_infer.infer_sum)
registry.register("argwhere", "mps", argwhere)
registry.register("baddbmm", "mps", baddbmm, meta=meta_infer.infer_binary)
registry.register("cummin", "mps", cummin)

# ---------------------------------------------------------------------------
# Category C2 gap-fill ops
# ---------------------------------------------------------------------------
from .ops import (
    diff,
    bincount,
    cdist,
    aminmax,
    quantile,
    nanquantile,
    nanmedian,
    histc,
    histogram,
    bucketize,
    isneginf,
    isposinf,
    isreal,
    isin,
    heaviside,
)

registry.register("diff", "mps", diff, meta=meta_infer.infer_unary)
registry.register("bincount", "mps", bincount)
registry.register("cdist", "mps", cdist)
registry.register("aminmax", "mps", aminmax)
registry.register("quantile", "mps", quantile)
registry.register("nanquantile", "mps", nanquantile)
registry.register("nanmedian", "mps", nanmedian)
registry.register("histc", "mps", histc)
registry.register("histogram", "mps", histogram)
registry.register("bucketize", "mps", bucketize)
registry.register("isneginf", "mps", isneginf, meta=meta_infer.infer_unary)
registry.register("isposinf", "mps", isposinf, meta=meta_infer.infer_unary)
registry.register("isreal", "mps", isreal, meta=meta_infer.infer_unary)
registry.register("isin", "mps", isin, meta=meta_infer.infer_binary)
registry.register("heaviside", "mps", heaviside, meta=meta_infer.infer_binary)

# ---------------------------------------------------------------------------
# Optimizer step ops
# ---------------------------------------------------------------------------
from .optim_ops import (
    _sgd_step,
    _adam_step,
    _adamw_step,
    _adagrad_step,
    _rmsprop_step,
    _adadelta_step,
    _adamax_step,
    _nadam_step,
    _radam_step,
    _asgd_step,
    _rprop_step,
    _sparse_adam_step,
)
registry.register("_sgd_step", "mps", _sgd_step)
registry.register("_adam_step", "mps", _adam_step)
registry.register("_adamw_step", "mps", _adamw_step)
registry.register("_adagrad_step", "mps", _adagrad_step)
registry.register("_rmsprop_step", "mps", _rmsprop_step)
registry.register("_adadelta_step", "mps", _adadelta_step)
registry.register("_adamax_step", "mps", _adamax_step)
registry.register("_nadam_step", "mps", _nadam_step)
registry.register("_radam_step", "mps", _radam_step)
registry.register("_asgd_step", "mps", _asgd_step)
registry.register("_rprop_step", "mps", _rprop_step)
registry.register("_sparse_adam_step", "mps", _sparse_adam_step)

# ---------------------------------------------------------------------------
# torch.linalg ops
# ---------------------------------------------------------------------------
from .ops import (
    linalg_cholesky,
    linalg_cond,
    linalg_det,
    linalg_eig,
    linalg_eigh,
    linalg_eigvals,
    linalg_eigvalsh,
    linalg_householder_product,
    linalg_inv,
    linalg_lstsq,
    linalg_lu,
    linalg_lu_factor,
    linalg_lu_solve,
    linalg_matrix_exp,
    linalg_matrix_norm,
    linalg_matrix_power,
    linalg_matrix_rank,
    linalg_multi_dot,
    linalg_norm,
    linalg_pinv,
    linalg_slogdet,
    linalg_solve,
    linalg_solve_triangular,
    linalg_svd,
    linalg_svdvals,
    linalg_tensorinv,
    linalg_tensorsolve,
    linalg_vander,
    linalg_vector_norm,
)

registry.register("linalg_cholesky", "mps", linalg_cholesky)
registry.register("linalg_cond", "mps", linalg_cond)
registry.register("linalg_det", "mps", linalg_det)
registry.register("linalg_eig", "mps", linalg_eig)
registry.register("linalg_eigh", "mps", linalg_eigh)
registry.register("linalg_eigvals", "mps", linalg_eigvals)
registry.register("linalg_eigvalsh", "mps", linalg_eigvalsh)
registry.register("linalg_householder_product", "mps", linalg_householder_product)
registry.register("linalg_inv", "mps", linalg_inv)
registry.register("linalg_lstsq", "mps", linalg_lstsq)
registry.register("linalg_lu", "mps", linalg_lu)
registry.register("linalg_lu_factor", "mps", linalg_lu_factor)
registry.register("linalg_lu_solve", "mps", linalg_lu_solve)
registry.register("linalg_matrix_exp", "mps", linalg_matrix_exp)
registry.register("linalg_matrix_norm", "mps", linalg_matrix_norm)
registry.register("linalg_matrix_power", "mps", linalg_matrix_power)
registry.register("linalg_matrix_rank", "mps", linalg_matrix_rank)
registry.register("linalg_multi_dot", "mps", linalg_multi_dot)
registry.register("linalg_norm", "mps", linalg_norm)
registry.register("linalg_pinv", "mps", linalg_pinv)
registry.register("linalg_slogdet", "mps", linalg_slogdet)
registry.register("linalg_solve", "mps", linalg_solve)
registry.register("linalg_solve_triangular", "mps", linalg_solve_triangular)
registry.register("linalg_svd", "mps", linalg_svd)
registry.register("linalg_svdvals", "mps", linalg_svdvals)
registry.register("linalg_tensorinv", "mps", linalg_tensorinv)
registry.register("linalg_tensorsolve", "mps", linalg_tensorsolve)
registry.register("linalg_vander", "mps", linalg_vander)
registry.register("linalg_vector_norm", "mps", linalg_vector_norm)

# ---------------------------------------------------------------------------
# torch.fft ops
# ---------------------------------------------------------------------------
from .ops import (
    fft_fft,
    fft_ifft,
    fft_fft2,
    fft_ifft2,
    fft_fftn,
    fft_ifftn,
    fft_rfft,
    fft_irfft,
    fft_rfft2,
    fft_irfft2,
    fft_rfftn,
    fft_irfftn,
    fft_hfft,
    fft_ihfft,
    fft_fftshift,
    fft_ifftshift,
)

registry.register("fft_fft", "mps", fft_fft)
registry.register("fft_ifft", "mps", fft_ifft)
registry.register("fft_fft2", "mps", fft_fft2)
registry.register("fft_ifft2", "mps", fft_ifft2)
registry.register("fft_fftn", "mps", fft_fftn)
registry.register("fft_ifftn", "mps", fft_ifftn)
registry.register("fft_rfft", "mps", fft_rfft)
registry.register("fft_irfft", "mps", fft_irfft)
registry.register("fft_rfft2", "mps", fft_rfft2)
registry.register("fft_irfft2", "mps", fft_irfft2)
registry.register("fft_rfftn", "mps", fft_rfftn)
registry.register("fft_irfftn", "mps", fft_irfftn)
registry.register("fft_hfft", "mps", fft_hfft)
registry.register("fft_ihfft", "mps", fft_ihfft)
registry.register("fft_fftshift", "mps", fft_fftshift)
registry.register("fft_ifftshift", "mps", fft_ifftshift)

# ---------------------------------------------------------------------------
# torch.special ops
# ---------------------------------------------------------------------------
from .ops import (
    special_digamma,
    special_entr,
    special_erfcx,
    special_erfinv,
    special_gammainc,
    special_gammaincc,
    special_gammaln,
    special_i0,
    special_i0e,
    special_i1,
    special_i1e,
    special_log_ndtr,
    special_logit,
    special_multigammaln,
    special_ndtr,
    special_ndtri,
    special_polygamma,
    special_sinc,
    special_xlog1py,
    special_xlogy,
    special_zeta,
)

registry.register("special_digamma", "mps", special_digamma)
registry.register("special_entr", "mps", special_entr)
registry.register("special_erfcx", "mps", special_erfcx)
registry.register("special_erfinv", "mps", special_erfinv)
registry.register("special_gammainc", "mps", special_gammainc)
registry.register("special_gammaincc", "mps", special_gammaincc)
registry.register("special_gammaln", "mps", special_gammaln)
registry.register("special_i0", "mps", special_i0)
registry.register("special_i0e", "mps", special_i0e)
registry.register("special_i1", "mps", special_i1)
registry.register("special_i1e", "mps", special_i1e)
registry.register("special_log_ndtr", "mps", special_log_ndtr)
registry.register("special_logit", "mps", special_logit)
registry.register("special_multigammaln", "mps", special_multigammaln)
registry.register("special_ndtr", "mps", special_ndtr)
registry.register("special_ndtri", "mps", special_ndtri)
registry.register("special_polygamma", "mps", special_polygamma)
registry.register("special_sinc", "mps", special_sinc)
registry.register("special_xlog1py", "mps", special_xlog1py)
registry.register("special_xlogy", "mps", special_xlogy)
registry.register("special_zeta", "mps", special_zeta)

# ---------------------------------------------------------------------------
# F.affine_grid / F.grid_sample
# ---------------------------------------------------------------------------
from .ops import grid_sample, affine_grid
registry.register("grid_sample", "mps", grid_sample)
registry.register("affine_grid", "mps", affine_grid)

# ---------------------------------------------------------------------------
# F.unfold (im2col) / F.fold (col2im)
# ---------------------------------------------------------------------------
from .ops import im2col, col2im
registry.register("im2col", "mps", im2col)
registry.register("col2im", "mps", col2im)

# ---------------------------------------------------------------------------
# one_hot
# ---------------------------------------------------------------------------
from .ops import one_hot
registry.register("one_hot", "mps", one_hot)

# ---------------------------------------------------------------------------
# uniform (out-of-place)
# ---------------------------------------------------------------------------
from .ops import uniform
registry.register("uniform", "mps", uniform)

# ---------------------------------------------------------------------------
# Upsample ops
# ---------------------------------------------------------------------------
from .ops import (
    upsample_nearest1d,
    upsample_linear1d,
    upsample_nearest2d,
    upsample_bilinear2d,
    upsample_bicubic2d,
)
registry.register("upsample_nearest1d", "mps", upsample_nearest1d)
registry.register("upsample_linear1d", "mps", upsample_linear1d)
registry.register("upsample_nearest2d", "mps", upsample_nearest2d)
registry.register("upsample_bilinear2d", "mps", upsample_bilinear2d)
registry.register("upsample_bicubic2d", "mps", upsample_bicubic2d)

# ---------------------------------------------------------------------------
# 1D pooling
# ---------------------------------------------------------------------------
from .ops import max_pool1d, avg_pool1d, adaptive_avg_pool1d
registry.register("max_pool1d", "mps", max_pool1d)
registry.register("avg_pool1d", "mps", avg_pool1d)
registry.register("adaptive_avg_pool1d", "mps", adaptive_avg_pool1d)

# ---------------------------------------------------------------------------
# 3D conv/pool
# ---------------------------------------------------------------------------
from .ops import conv_transpose3d, max_pool3d, avg_pool3d, adaptive_avg_pool3d, conv3d
registry.register("conv3d", "mps", conv3d)
registry.register("conv_transpose3d", "mps", conv_transpose3d)
registry.register("max_pool3d", "mps", max_pool3d)
registry.register("avg_pool3d", "mps", avg_pool3d)
registry.register("adaptive_avg_pool3d", "mps", adaptive_avg_pool3d)

# ---------------------------------------------------------------------------
# addmm
# ---------------------------------------------------------------------------
from .ops import addmm
registry.register("addmm", "mps", addmm)

# ---------------------------------------------------------------------------
# adaptive_max_pool
# ---------------------------------------------------------------------------
from .ops import adaptive_max_pool2d, adaptive_max_pool1d
registry.register("adaptive_max_pool2d", "mps", adaptive_max_pool2d)
registry.register("adaptive_max_pool1d", "mps", adaptive_max_pool1d)

# ---------------------------------------------------------------------------
# Activation ops
# ---------------------------------------------------------------------------
from .ops import selu, celu, threshold, hardshrink, softshrink, rrelu
registry.register("selu", "mps", selu, meta=meta_infer.infer_unary)
registry.register("celu", "mps", celu, meta=meta_infer.infer_unary)
registry.register("threshold", "mps", threshold, meta=meta_infer.infer_unary)
registry.register("hardshrink", "mps", hardshrink, meta=meta_infer.infer_unary)
registry.register("softshrink", "mps", softshrink, meta=meta_infer.infer_unary)
registry.register("rrelu", "mps", rrelu, meta=meta_infer.infer_unary)

# ---------------------------------------------------------------------------
# CTC loss
# ---------------------------------------------------------------------------
from .ops import ctc_loss
registry.register("ctc_loss", "mps", ctc_loss)

# ---------------------------------------------------------------------------
# Round 12 ops
# ---------------------------------------------------------------------------
from .ops import hardswish, hardsigmoid, softsign, normalize
registry.register("hardswish", "mps", hardswish, meta=meta_infer.infer_unary)
registry.register("hardsigmoid", "mps", hardsigmoid, meta=meta_infer.infer_unary)
registry.register("softsign", "mps", softsign, meta=meta_infer.infer_unary)
registry.register("normalize", "mps", normalize, meta=meta_infer.infer_unary)
registry.register("moveaxis", "mps", movedim, meta=meta_infer.infer_movedim)
registry.register("min_", "mps", min_, meta=meta_infer.infer_binary)
registry.register("max_", "mps", max_, meta=meta_infer.infer_binary)
