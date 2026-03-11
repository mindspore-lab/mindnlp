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
    randn_create,
    rand_create,
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

registry.register("add", "cpu", add, meta=meta_infer.infer_binary)
registry.register("mul", "cpu", mul, meta=meta_infer.infer_binary)
registry.register("div", "cpu", div, meta=meta_infer.infer_binary)
registry.register("true_divide", "cpu", true_divide, meta=meta_infer.infer_binary)
registry.register("matmul", "cpu", matmul, meta=meta_infer.infer_matmul)
registry.register("mm", "cpu", mm, meta=meta_infer.infer_matmul)
registry.register("bmm", "cpu", bmm, meta=meta_infer.infer_matmul)
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
registry.register("square", "cpu", square, meta=meta_infer.infer_unary)
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
registry.register("silu", "cpu", silu, meta=meta_infer.infer_unary)
registry.register("leaky_relu", "cpu", leaky_relu, meta=meta_infer.infer_unary)
registry.register("elu", "cpu", elu, meta=meta_infer.infer_unary)
registry.register("mish", "cpu", mish, meta=meta_infer.infer_unary)
registry.register("prelu", "cpu", prelu, meta=meta_infer.infer_binary)
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
registry.register("flip", "cpu", flip, meta=meta_infer.infer_flip)
registry.register("roll", "cpu", roll, meta=meta_infer.infer_roll)
registry.register("rot90", "cpu", rot90, meta=meta_infer.infer_rot90)
registry.register("repeat", "cpu", repeat, meta=meta_infer.infer_repeat)
registry.register("repeat_interleave", "cpu", repeat_interleave, meta=meta_infer.infer_repeat_interleave)
registry.register("tile", "cpu", tile, meta=meta_infer.infer_tile)
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
registry.register("mean", "cpu", mean_, meta=meta_infer.infer_sum)
registry.register("std", "cpu", std_, meta=meta_infer.infer_sum)
registry.register("all", "cpu", all_, meta=meta_infer.infer_reduce_bool)
registry.register("batch_norm", "cpu", batch_norm, meta=meta_infer.infer_unary)
registry.register("instance_norm", "cpu", instance_norm, meta=meta_infer.infer_unary)
registry.register("group_norm", "cpu", group_norm, meta=meta_infer.infer_unary)
registry.register("layer_norm", "cpu", layer_norm, meta=meta_infer.infer_unary)
registry.register("dropout", "cpu", dropout, meta=meta_infer.infer_unary)
registry.register("gelu", "cpu", gelu, meta=meta_infer.infer_unary)
registry.register("pad", "cpu", pad, meta=meta_infer.infer_unary)
registry.register("softmax", "cpu", softmax, meta=meta_infer.infer_unary)
registry.register("log_softmax", "cpu", log_softmax, meta=meta_infer.infer_unary)
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
registry.register("eq", "cpu", eq, meta=meta_infer.infer_binary_bool)
registry.register("ne", "cpu", ne, meta=meta_infer.infer_binary_bool)
registry.register("lt", "cpu", lt, meta=meta_infer.infer_binary_bool)
registry.register("le", "cpu", le, meta=meta_infer.infer_binary_bool)
registry.register("gt", "cpu", gt, meta=meta_infer.infer_binary_bool)
registry.register("ge", "cpu", ge, meta=meta_infer.infer_binary_bool)
registry.register("add_", "cpu", add_, meta=meta_infer.infer_binary)
registry.register("mul_", "cpu", mul_, meta=meta_infer.infer_binary)
registry.register("relu_", "cpu", relu_, meta=meta_infer.infer_unary)
registry.register("zero_", "cpu", zero_, meta=meta_infer.infer_unary)
registry.register("uniform_", "cpu", uniform_, meta=meta_infer.infer_unary)
registry.register("normal_", "cpu", normal_, meta=meta_infer.infer_unary)
registry.register("bernoulli_", "cpu", bernoulli_, meta=meta_infer.infer_unary)
registry.register("exponential_", "cpu", exponential_, meta=meta_infer.infer_unary)
registry.register("log_normal_", "cpu", log_normal_, meta=meta_infer.infer_unary)
registry.register("cauchy_", "cpu", cauchy_, meta=meta_infer.infer_unary)
registry.register("geometric_", "cpu", geometric_, meta=meta_infer.infer_unary)
registry.register("fill_", "cpu", fill_, meta=meta_infer.infer_unary)
registry.register("clamp_", "cpu", clamp_, meta=meta_infer.infer_unary)
registry.register("copy_", "cpu", copy_, meta=meta_infer.infer_unary)
registry.register("erfinv_", "cpu", erfinv_, meta=meta_infer.infer_unary)
registry.register("sub_", "cpu", sub_, meta=meta_infer.infer_binary)
registry.register("reshape", "cpu", view_backend.reshape, meta=meta_infer.infer_view)
registry.register("view", "cpu", view_backend.view, meta=meta_infer.infer_view)
registry.register("transpose", "cpu", view_backend.transpose, meta=meta_infer.infer_transpose)
registry.register("squeeze", "cpu", view_backend.squeeze, meta=meta_infer.infer_view)
registry.register("unsqueeze", "cpu", view_backend.unsqueeze, meta=meta_infer.infer_view)
registry.register("permute", "cpu", view_backend.permute, meta=meta_infer.infer_view)
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
registry.register("randn", "cpu", randn_create)
registry.register("rand", "cpu", rand_create)
registry.register("randint", "cpu", randint_create)
registry.register("randperm", "cpu", randperm_create)
registry.register("linalg_qr", "cpu", linalg_qr)

# Tensor indexing / selection ops
registry.register("narrow", "cpu", narrow)
registry.register("select", "cpu", select)
registry.register("expand", "cpu", expand)
registry.register("masked_fill", "cpu", masked_fill, meta=meta_infer.infer_unary)
registry.register("masked_fill_", "cpu", masked_fill_, meta=meta_infer.infer_unary)
registry.register("index_put_", "cpu", index_put_)
registry.register("index_put", "cpu", index_put)
registry.register("index_copy_", "cpu", index_copy_)
registry.register("index_fill_", "cpu", index_fill_)
registry.register("index_add_", "cpu", index_add_)
registry.register("scatter_", "cpu", scatter_)
registry.register("scatter_add_", "cpu", scatter_add_)
registry.register("masked_scatter_", "cpu", masked_scatter_)
registry.register("unfold", "cpu", unfold)

registry.register("var", "cpu", var_, meta=meta_infer.infer_sum)
registry.register("norm", "cpu", norm_, meta=meta_infer.infer_sum)
registry.register("prod", "cpu", prod_, meta=meta_infer.infer_sum)
registry.register("floor_divide", "cpu", floor_divide, meta=meta_infer.infer_binary)
registry.register("rms_norm", "cpu", rms_norm, meta=meta_infer.infer_unary)

# Conv operations
registry.register("conv2d", "cpu", conv2d)
registry.register("conv1d", "cpu", conv1d)
registry.register("conv_transpose2d", "cpu", conv_transpose2d)
registry.register("conv_transpose1d", "cpu", conv_transpose1d)

# Pooling operations
registry.register("max_pool2d", "cpu", max_pool2d)
registry.register("avg_pool2d", "cpu", avg_pool2d)
registry.register("adaptive_avg_pool2d", "cpu", adaptive_avg_pool2d)

registry.register("embedding", "cpu", embedding)

# ---------------------------------------------------------------------------
# Registrations for new ops
# ---------------------------------------------------------------------------

# Math ops
registry.register("sub", "cpu", sub, meta=meta_infer.infer_binary)
registry.register("log1p", "cpu", log1p, meta=meta_infer.infer_unary)
registry.register("expm1", "cpu", expm1, meta=meta_infer.infer_unary)
registry.register("reciprocal", "cpu", reciprocal, meta=meta_infer.infer_unary)
registry.register("maximum", "cpu", maximum, meta=meta_infer.infer_binary)
registry.register("minimum", "cpu", minimum, meta=meta_infer.infer_binary)
registry.register("dot", "cpu", dot, meta=meta_infer.infer_dot)
registry.register("outer", "cpu", outer, meta=meta_infer.infer_outer)
registry.register("inner", "cpu", inner, meta=meta_infer.infer_binary)
registry.register("mv", "cpu", mv, meta=meta_infer.infer_binary)
registry.register("cross", "cpu", cross, meta=meta_infer.infer_binary)
registry.register("tensordot", "cpu", tensordot)
registry.register("einsum", "cpu", einsum)

# Logical ops
registry.register("logical_and", "cpu", logical_and, meta=meta_infer.infer_binary_bool)
registry.register("logical_or", "cpu", logical_or, meta=meta_infer.infer_binary_bool)
registry.register("logical_not", "cpu", logical_not, meta=meta_infer.infer_unary_bool)

# In-place ops
registry.register("div_", "cpu", div_, meta=meta_infer.infer_binary)

registry.register("logical_xor", "cpu", logical_xor, meta=meta_infer.infer_binary_bool)

# Bitwise ops
registry.register("bitwise_and", "cpu", bitwise_and, meta=meta_infer.infer_binary)
registry.register("bitwise_or", "cpu", bitwise_or, meta=meta_infer.infer_binary)
registry.register("bitwise_xor", "cpu", bitwise_xor, meta=meta_infer.infer_binary)
registry.register("bitwise_not", "cpu", bitwise_not, meta=meta_infer.infer_unary)

# Random in-place op
registry.register("randint_", "cpu", randint_, meta=meta_infer.infer_unary)
registry.register("random_", "cpu", random_, meta=meta_infer.infer_unary)

# Shape ops
registry.register("flatten", "cpu", flatten, meta=meta_infer.infer_flatten)
registry.register("unflatten", "cpu", unflatten, meta=meta_infer.infer_unflatten)
registry.register("broadcast_to", "cpu", broadcast_to, meta=meta_infer.infer_broadcast_to)
registry.register("movedim", "cpu", movedim, meta=meta_infer.infer_movedim)
registry.register("diagonal", "cpu", diagonal, meta=meta_infer.infer_diagonal)

# Search ops (no meta — output shape is data-dependent or returns tuples)
registry.register("unique", "cpu", unique)
registry.register("searchsorted", "cpu", searchsorted)
registry.register("kthvalue", "cpu", kthvalue)
registry.register("median", "cpu", median)

# New GROUP C ops
registry.register("logsumexp", "cpu", logsumexp, meta=meta_infer.infer_sum)
registry.register("trace", "cpu", trace)  # Returns scalar
registry.register("det", "cpu", det)  # Returns scalar or batch of scalars
registry.register("matrix_power", "cpu", matrix_power, meta=meta_infer.infer_unary)
registry.register("dist", "cpu", dist)  # Returns scalar
registry.register("renorm", "cpu", renorm, meta=meta_infer.infer_unary)
registry.register("nansum", "cpu", nansum, meta=meta_infer.infer_sum)
registry.register("nanmean", "cpu", nanmean, meta=meta_infer.infer_sum)
registry.register("argwhere", "cpu", argwhere)  # Returns 2D tensor
registry.register("baddbmm", "cpu", baddbmm, meta=meta_infer.infer_binary)
registry.register("cummin", "cpu", cummin)  # Returns namedtuple

# Top-level gap-fill ops (Category C2)
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

registry.register("diff", "cpu", diff, meta=meta_infer.infer_unary)
registry.register("bincount", "cpu", bincount)
registry.register("cdist", "cpu", cdist)
registry.register("aminmax", "cpu", aminmax)
registry.register("quantile", "cpu", quantile)
registry.register("nanquantile", "cpu", nanquantile)
registry.register("nanmedian", "cpu", nanmedian)
registry.register("histc", "cpu", histc)
registry.register("histogram", "cpu", histogram)
registry.register("bucketize", "cpu", bucketize)
registry.register("isneginf", "cpu", isneginf, meta=meta_infer.infer_unary)
registry.register("isposinf", "cpu", isposinf, meta=meta_infer.infer_unary)
registry.register("isreal", "cpu", isreal, meta=meta_infer.infer_unary)
registry.register("isin", "cpu", isin, meta=meta_infer.infer_binary)
registry.register("heaviside", "cpu", heaviside, meta=meta_infer.infer_binary)

# Optimizer step ops
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
registry.register("_sgd_step", "cpu", _sgd_step)
registry.register("_adam_step", "cpu", _adam_step)
registry.register("_adamw_step", "cpu", _adamw_step)
registry.register("_adagrad_step", "cpu", _adagrad_step)
registry.register("_rmsprop_step", "cpu", _rmsprop_step)
registry.register("_adadelta_step", "cpu", _adadelta_step)
registry.register("_adamax_step", "cpu", _adamax_step)
registry.register("_nadam_step", "cpu", _nadam_step)
registry.register("_radam_step", "cpu", _radam_step)
registry.register("_asgd_step", "cpu", _asgd_step)
registry.register("_rprop_step", "cpu", _rprop_step)
registry.register("_sparse_adam_step", "cpu", _sparse_adam_step)

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

registry.register("linalg_cholesky", "cpu", linalg_cholesky)
registry.register("linalg_cond", "cpu", linalg_cond)
registry.register("linalg_det", "cpu", linalg_det)
registry.register("linalg_eig", "cpu", linalg_eig)
registry.register("linalg_eigh", "cpu", linalg_eigh)
registry.register("linalg_eigvals", "cpu", linalg_eigvals)
registry.register("linalg_eigvalsh", "cpu", linalg_eigvalsh)
registry.register("linalg_householder_product", "cpu", linalg_householder_product)
registry.register("linalg_inv", "cpu", linalg_inv)
registry.register("linalg_lstsq", "cpu", linalg_lstsq)
registry.register("linalg_lu", "cpu", linalg_lu)
registry.register("linalg_lu_factor", "cpu", linalg_lu_factor)
registry.register("linalg_lu_solve", "cpu", linalg_lu_solve)
registry.register("linalg_matrix_exp", "cpu", linalg_matrix_exp)
registry.register("linalg_matrix_norm", "cpu", linalg_matrix_norm)
registry.register("linalg_matrix_power", "cpu", linalg_matrix_power)
registry.register("linalg_matrix_rank", "cpu", linalg_matrix_rank)
registry.register("linalg_multi_dot", "cpu", linalg_multi_dot)
registry.register("linalg_norm", "cpu", linalg_norm)
registry.register("linalg_pinv", "cpu", linalg_pinv)
registry.register("linalg_slogdet", "cpu", linalg_slogdet)
registry.register("linalg_solve", "cpu", linalg_solve)
registry.register("linalg_solve_triangular", "cpu", linalg_solve_triangular)
registry.register("linalg_svd", "cpu", linalg_svd)
registry.register("linalg_svdvals", "cpu", linalg_svdvals)
registry.register("linalg_tensorinv", "cpu", linalg_tensorinv)
registry.register("linalg_tensorsolve", "cpu", linalg_tensorsolve)
registry.register("linalg_vander", "cpu", linalg_vander)
registry.register("linalg_vector_norm", "cpu", linalg_vector_norm)

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

registry.register("fft_fft", "cpu", fft_fft)
registry.register("fft_ifft", "cpu", fft_ifft)
registry.register("fft_fft2", "cpu", fft_fft2)
registry.register("fft_ifft2", "cpu", fft_ifft2)
registry.register("fft_fftn", "cpu", fft_fftn)
registry.register("fft_ifftn", "cpu", fft_ifftn)
registry.register("fft_rfft", "cpu", fft_rfft)
registry.register("fft_irfft", "cpu", fft_irfft)
registry.register("fft_rfft2", "cpu", fft_rfft2)
registry.register("fft_irfft2", "cpu", fft_irfft2)
registry.register("fft_rfftn", "cpu", fft_rfftn)
registry.register("fft_irfftn", "cpu", fft_irfftn)
registry.register("fft_hfft", "cpu", fft_hfft)
registry.register("fft_ihfft", "cpu", fft_ihfft)
registry.register("fft_fftshift", "cpu", fft_fftshift)
registry.register("fft_ifftshift", "cpu", fft_ifftshift)

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

registry.register("special_digamma", "cpu", special_digamma)
registry.register("special_entr", "cpu", special_entr)
registry.register("special_erfcx", "cpu", special_erfcx)
registry.register("special_erfinv", "cpu", special_erfinv)
registry.register("special_gammainc", "cpu", special_gammainc)
registry.register("special_gammaincc", "cpu", special_gammaincc)
registry.register("special_gammaln", "cpu", special_gammaln)
registry.register("special_i0", "cpu", special_i0)
registry.register("special_i0e", "cpu", special_i0e)
registry.register("special_i1", "cpu", special_i1)
registry.register("special_i1e", "cpu", special_i1e)
registry.register("special_log_ndtr", "cpu", special_log_ndtr)
registry.register("special_logit", "cpu", special_logit)
registry.register("special_multigammaln", "cpu", special_multigammaln)
registry.register("special_ndtr", "cpu", special_ndtr)
registry.register("special_ndtri", "cpu", special_ndtri)
registry.register("special_polygamma", "cpu", special_polygamma)
registry.register("special_sinc", "cpu", special_sinc)
registry.register("special_xlog1py", "cpu", special_xlog1py)
registry.register("special_xlogy", "cpu", special_xlogy)
registry.register("special_zeta", "cpu", special_zeta)

# ---------------------------------------------------------------------------
# F.affine_grid / F.grid_sample
# ---------------------------------------------------------------------------
from .ops import grid_sample, affine_grid
registry.register("grid_sample", "cpu", grid_sample)
registry.register("affine_grid", "cpu", affine_grid)

# ---------------------------------------------------------------------------
# F.unfold (im2col) / F.fold (col2im)
# ---------------------------------------------------------------------------
from .ops import im2col, col2im
registry.register("im2col", "cpu", im2col)
registry.register("col2im", "cpu", col2im)

# ---------------------------------------------------------------------------
# one_hot (was missing registration)
# ---------------------------------------------------------------------------
from .ops import one_hot
registry.register("one_hot", "cpu", one_hot)

# ---------------------------------------------------------------------------
# uniform (out-of-place)
# ---------------------------------------------------------------------------
from .ops import uniform
registry.register("uniform", "cpu", uniform)

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
registry.register("upsample_nearest1d", "cpu", upsample_nearest1d)
registry.register("upsample_linear1d", "cpu", upsample_linear1d)
registry.register("upsample_nearest2d", "cpu", upsample_nearest2d)
registry.register("upsample_bilinear2d", "cpu", upsample_bilinear2d)
registry.register("upsample_bicubic2d", "cpu", upsample_bicubic2d)

# 1D pooling ops
from .ops import max_pool1d, avg_pool1d, adaptive_avg_pool1d
registry.register("max_pool1d", "cpu", max_pool1d)
registry.register("avg_pool1d", "cpu", avg_pool1d)
registry.register("adaptive_avg_pool1d", "cpu", adaptive_avg_pool1d)

# 3D conv/pool ops
from .ops import conv_transpose3d, max_pool3d, avg_pool3d, adaptive_avg_pool3d, conv3d
registry.register("conv3d", "cpu", conv3d)
registry.register("conv_transpose3d", "cpu", conv_transpose3d)
registry.register("max_pool3d", "cpu", max_pool3d)
registry.register("avg_pool3d", "cpu", avg_pool3d)
registry.register("adaptive_avg_pool3d", "cpu", adaptive_avg_pool3d)

# addmm
from .ops import addmm
registry.register("addmm", "cpu", addmm)

# adaptive_max_pool ops
from .ops import adaptive_max_pool2d, adaptive_max_pool1d
registry.register("adaptive_max_pool2d", "cpu", adaptive_max_pool2d)
registry.register("adaptive_max_pool1d", "cpu", adaptive_max_pool1d)

# Round 6: missing activation CPU forwards
from .ops import selu, celu, threshold, hardshrink, softshrink, rrelu
registry.register("selu", "cpu", selu, meta=meta_infer.infer_unary)
registry.register("celu", "cpu", celu, meta=meta_infer.infer_unary)
registry.register("threshold", "cpu", threshold, meta=meta_infer.infer_unary)
registry.register("hardshrink", "cpu", hardshrink, meta=meta_infer.infer_unary)
registry.register("softshrink", "cpu", softshrink, meta=meta_infer.infer_unary)
registry.register("rrelu", "cpu", rrelu, meta=meta_infer.infer_unary)

# CTC loss
from .ops import ctc_loss
registry.register("ctc_loss", "cpu", ctc_loss)

# Round 12: remaining CPU forward gaps
from .ops import hardswish, hardsigmoid, softsign, normalize, movedim, min_, max_
registry.register("hardswish", "cpu", hardswish, meta=meta_infer.infer_unary)
registry.register("hardsigmoid", "cpu", hardsigmoid, meta=meta_infer.infer_unary)
registry.register("softsign", "cpu", softsign, meta=meta_infer.infer_unary)
registry.register("normalize", "cpu", normalize, meta=meta_infer.infer_unary)
registry.register("moveaxis", "cpu", movedim, meta=meta_infer.infer_movedim)
registry.register("min_", "cpu", min_, meta=meta_infer.infer_binary)
registry.register("max_", "cpu", max_, meta=meta_infer.infer_binary)
