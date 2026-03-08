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
    isneginf,
    isposinf,
    log,
    log10,
    log2,
    matmul,
    mul,
    neg,
    relu6,
    sign,
    signbit,
    square,
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
    uniform_,
    normal_,
    randint_,
    random_,
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
    getitem,
    setitem,
    all_,
    any_,
    count_nonzero,
    flip,
    roll,
    nonzero,
    cumsum,
    cumprod,
    cummax,
    argsort,
    sort,
    topk,
    tril,
    triu,
    rot90,
    repeat,
    repeat_interleave,
    tile,
    scatter,
    tril_indices,
    triu_indices,
    diag,
    cartesian_prod,
    block_diag,
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
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
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
    silu,
    leaky_relu,
    elu,
    mish,
    prelu,
    batch_norm,
    instance_norm,
    group_norm,
    gather,
    index_select,
    take,
    take_along_dim,
    masked_select,
    dropout,
    pad,
    pad_sequence,
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
    max_pool3d,
    avg_pool2d,
    adaptive_avg_pool2d,
    adaptive_max_pool2d,
    # P1 ops
    std_,
    reciprocal_,
    addmm,
    einsum_,
    upsample_nearest2d,
    upsample_bilinear2d,
    one_hot,
    # Logical ops
    logical_and,
    logical_or,
    logical_not,
    # Bitwise ops
    bitwise_not,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    # Math ops
    expm1,
    log1p,
    # Element-wise min/max
    maximum,
    minimum,
    # Linear algebra
    dot,
    mv,
    outer,
    # Reduction ops
    median,
    kthvalue,
    # Search / unique
    searchsorted,
    unique,
    # Random
    randperm,
    # Shape
    flatten_op,
    # P1 new ops
    baddbmm,
    trace_op,
    cummin_op,
    logsumexp_op,
    renorm_op,
    logical_xor,
    # P2 new ops
    nansum,
    cross_op,
    # P0: ACLNN large kernel ops
    im2col_op,
    grid_sample_op,
    affine_grid_op,
    # P1: View/reshape ops
    broadcast_to_op,
    movedim_op,
    unflatten_op,
    diagonal_op,
    # Missing forward ops — composites
    aminmax_op,
    nanmean_op,
    argwhere_op,
    det_op,
    diff_op,
    dist_op,
    heaviside_op,
    inner_op,
    tensordot_op,
    cdist_op,
    uniform_op,
    isreal_op,
    isin_op,
    bincount_op,
    bucketize_op,
    histc_op,
    histogram_op,
    quantile_op,
    nanquantile_op,
    nanmedian_op,
    matrix_power_op,
    col2im_op,
    # Phase 1: ACLNN large kernel ops (910B confirmed)
    special_digamma,
    special_erfinv,
    special_gammaln,
    special_sinc,
    linalg_inv,
    mm_op,
    bmm_op,
    linalg_vector_norm_op,
    aminmax_aclnn,
    bincount_aclnn,
    adaptive_avg_pool3d_op,
    upsample_bicubic2d_op,
    upsample_linear1d_op,
    _adam_step_op,
    _adamw_step_op,
    # Phase 2: activation composites
    selu_op,
    celu_op,
    threshold_op,
    hardshrink_op,
    softshrink_op,
    hardswish_op,
    hardsigmoid_op,
    softsign_op,
    rrelu_op,
    normalize_op,
    moveaxis_op,
    # Phase 3: 1D pooling composites
    adaptive_avg_pool1d_op,
    avg_pool1d_op,
    max_pool1d_op,
    adaptive_max_pool1d_op,
    # Phase 4: optimizer composites
    _sgd_step_op,
    _adagrad_step_op,
    _rmsprop_step_op,
    _adadelta_step_op,
    _adamax_step_op,
    _asgd_step_op,
    _nadam_step_op,
    _radam_step_op,
    _rprop_step_op,
    _sparse_adam_step_op,
    # Phase 5: special function composites + CPU fallbacks
    special_entr_op,
    special_erfcx_op,
    special_logit_op,
    special_ndtr_op,
    special_log_ndtr_op,
    special_xlogy_op,
    special_xlog1py_op,
    special_multigammaln_op,
    special_gammainc_op,
    special_gammaincc_op,
    special_i0_op,
    special_i0e_op,
    special_i1_op,
    special_i1e_op,
    special_ndtri_op,
    special_polygamma_op,
    special_zeta_op,
    # Phase 6: linalg composites + CPU fallbacks
    linalg_norm_op,
    linalg_matrix_norm_op,
    linalg_multi_dot_op,
    linalg_matrix_power_op,
    linalg_vander_op,
    linalg_cholesky_op,
    linalg_cond_op,
    linalg_det_op,
    linalg_eig_op,
    linalg_eigh_op,
    linalg_eigvals_op,
    linalg_eigvalsh_op,
    linalg_householder_product_op,
    linalg_lstsq_op,
    linalg_lu_op,
    linalg_lu_factor_op,
    linalg_lu_solve_op,
    linalg_matrix_exp_op,
    linalg_matrix_rank_op,
    linalg_pinv_op,
    linalg_slogdet_op,
    linalg_solve_op,
    linalg_solve_triangular_op,
    linalg_svd_op,
    linalg_svdvals_op,
    linalg_tensorinv_op,
    linalg_tensorsolve_op,
    # Phase 7: FFT CPU fallbacks
    fft_fft_op,
    fft_ifft_op,
    fft_rfft_op,
    fft_irfft_op,
    fft_fft2_op,
    fft_ifft2_op,
    fft_rfft2_op,
    fft_irfft2_op,
    fft_fftn_op,
    fft_ifftn_op,
    fft_rfftn_op,
    fft_irfftn_op,
    fft_hfft_op,
    fft_ihfft_op,
    fft_fftshift_op,
    fft_ifftshift_op,
    # Other missing ops
    conv3d_op,
    conv_transpose3d_op,
    avg_pool3d_op,
    upsample_nearest1d_op,
    ctc_loss_op,
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
registry.register("flip", "npu", flip, meta=meta_infer.infer_flip)
registry.register("roll", "npu", roll, meta=meta_infer.infer_roll)
registry.register("nonzero", "npu", nonzero, meta=meta_infer.infer_nonzero)
registry.register("cumsum", "npu", cumsum, meta=meta_infer.infer_unary)
registry.register("cumprod", "npu", cumprod, meta=meta_infer.infer_unary)
registry.register("cummax", "npu", cummax, meta=meta_infer.infer_cummax)
registry.register("argsort", "npu", argsort, meta=meta_infer.infer_argsort)
registry.register("sort", "npu", sort, meta=meta_infer.infer_sort)
registry.register("topk", "npu", topk, meta=meta_infer.infer_topk)
registry.register("tril", "npu", tril, meta=meta_infer.infer_unary)
registry.register("triu", "npu", triu, meta=meta_infer.infer_unary)
registry.register("rot90", "npu", rot90, meta=meta_infer.infer_rot90)
registry.register("repeat", "npu", repeat, meta=meta_infer.infer_repeat)
registry.register("repeat_interleave", "npu", repeat_interleave, meta=meta_infer.infer_repeat_interleave)
registry.register("tile", "npu", tile, meta=meta_infer.infer_tile)
registry.register("scatter", "npu", scatter, meta=meta_infer.infer_scatter)
registry.register("tril_indices", "npu", tril_indices, meta=meta_infer.infer_tril_indices)
registry.register("triu_indices", "npu", triu_indices, meta=meta_infer.infer_triu_indices)
registry.register("diag", "npu", diag, meta=meta_infer.infer_diag)
registry.register("cartesian_prod", "npu", cartesian_prod, meta=meta_infer.infer_cartesian_prod)
registry.register("block_diag", "npu", block_diag, meta=meta_infer.infer_block_diag)
registry.register("abs", "npu", abs, meta=meta_infer.infer_unary)
registry.register("neg", "npu", neg, meta=meta_infer.infer_unary)
registry.register("sign", "npu", sign, meta=meta_infer.infer_unary)
registry.register("signbit", "npu", signbit, meta=meta_infer.infer_unary_bool)
registry.register("square", "npu", square, meta=meta_infer.infer_unary)
registry.register("isfinite", "npu", isfinite, meta=meta_infer.infer_unary_bool)
registry.register("isinf", "npu", isinf, meta=meta_infer.infer_unary_bool)
registry.register("isnan", "npu", isnan, meta=meta_infer.infer_unary_bool)
registry.register("isneginf", "npu", isneginf, meta=meta_infer.infer_unary_bool)
registry.register("isposinf", "npu", isposinf, meta=meta_infer.infer_unary_bool)
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
# Element-wise min/max
registry.register("maximum", "npu", maximum, meta=meta_infer.infer_binary)
registry.register("minimum", "npu", minimum, meta=meta_infer.infer_binary)

registry.register("amin", "npu", amin, meta=meta_infer.infer_sum)
registry.register("amax", "npu", amax, meta=meta_infer.infer_sum)
registry.register("argmax", "npu", argmax, meta=meta_infer.infer_argmax)
registry.register("argmin", "npu", argmin, meta=meta_infer.infer_argmax)
# Reduction ops with dim
registry.register("median", "npu", median, meta=meta_infer.infer_sum)
registry.register("kthvalue", "npu", kthvalue, meta=meta_infer.infer_sum)
registry.register("add_", "npu", add_, meta=meta_infer.infer_binary)
registry.register("mul_", "npu", mul_, meta=meta_infer.infer_binary)
registry.register("relu_", "npu", relu_, meta=meta_infer.infer_unary)
registry.register("zero_", "npu", zero_, meta=meta_infer.infer_unary)
registry.register("uniform_", "npu", uniform_, meta=meta_infer.infer_unary)
registry.register("normal_", "npu", normal_, meta=meta_infer.infer_unary)
registry.register("randint_", "npu", randint_, meta=meta_infer.infer_unary)
registry.register("random_", "npu", random_, meta=meta_infer.infer_unary)
registry.register("bernoulli_", "npu", bernoulli_, meta=meta_infer.infer_unary)
registry.register("exponential_", "npu", exponential_, meta=meta_infer.infer_unary)
registry.register("log_normal_", "npu", log_normal_, meta=meta_infer.infer_unary)
registry.register("cauchy_", "npu", cauchy_, meta=meta_infer.infer_unary)
registry.register("geometric_", "npu", geometric_, meta=meta_infer.infer_unary)
registry.register("fill_", "npu", fill_, meta=meta_infer.infer_unary)
registry.register("clamp_", "npu", clamp_, meta=meta_infer.infer_unary)
registry.register("copy_", "npu", copy_, meta=meta_infer.infer_unary)
registry.register("erfinv_", "npu", erfinv_, meta=meta_infer.infer_unary)
registry.register("sub_", "npu", sub_, meta=meta_infer.infer_binary)
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
# Bitwise ops
registry.register("bitwise_not", "npu", bitwise_not, meta=meta_infer.infer_unary)
registry.register("bitwise_and", "npu", bitwise_and, meta=meta_infer.infer_binary)
registry.register("bitwise_or", "npu", bitwise_or, meta=meta_infer.infer_binary)
registry.register("bitwise_xor", "npu", bitwise_xor, meta=meta_infer.infer_binary)
# Math ops
registry.register("expm1", "npu", expm1, meta=meta_infer.infer_unary)
registry.register("log1p", "npu", log1p, meta=meta_infer.infer_unary)
registry.register("dot", "npu", dot, meta=meta_infer.infer_binary)
registry.register("mv", "npu", mv, meta=meta_infer.infer_matmul)
registry.register("outer", "npu", outer, meta=meta_infer.infer_binary)
registry.register("searchsorted", "npu", searchsorted, meta=meta_infer.infer_unary)
registry.register("unique", "npu", unique)
registry.register("randperm", "npu", randperm)
registry.register("flatten", "npu", flatten_op, meta=meta_infer.infer_view)
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
registry.register("eq", "npu", eq, meta=meta_infer.infer_binary_bool)
registry.register("ne", "npu", ne, meta=meta_infer.infer_binary_bool)
registry.register("lt", "npu", lt, meta=meta_infer.infer_binary_bool)
registry.register("le", "npu", le, meta=meta_infer.infer_binary_bool)
registry.register("gt", "npu", gt, meta=meta_infer.infer_binary_bool)
registry.register("ge", "npu", ge, meta=meta_infer.infer_binary_bool)
registry.register("reshape", "npu", view_backend.reshape, meta=meta_infer.infer_view)
registry.register("view", "npu", view_backend.view, meta=meta_infer.infer_view)
registry.register("transpose", "npu", view_backend.transpose, meta=meta_infer.infer_transpose)
registry.register("squeeze", "npu", view_backend.squeeze, meta=meta_infer.infer_view)
registry.register("unsqueeze", "npu", view_backend.unsqueeze, meta=meta_infer.infer_view)
registry.register("permute", "npu", view_backend.permute, meta=meta_infer.infer_view)
registry.register("to", "npu", convert_backend.to_device)

registry.register("tensor", "npu", tensor_create)
registry.register("zeros", "npu", zeros_create)
registry.register("ones", "npu", ones_create)
registry.register("empty", "npu", empty_create)
registry.register("randn", "npu", randn_create)
registry.register("rand", "npu", rand_create)
registry.register("randint", "npu", randint_create)
registry.register("randperm", "npu", randperm_create)
registry.register("arange", "npu", arange_create)
registry.register("linspace", "npu", linspace_create)
registry.register("full", "npu", full_create)
registry.register("logspace", "npu", logspace_create)
registry.register("eye", "npu", eye_create)
registry.register("range", "npu", range_create)
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
registry.register("gather", "npu", gather, meta=meta_infer.infer_gather)
registry.register("index_select", "npu", index_select, meta=meta_infer.infer_index_select)
registry.register("take", "npu", take, meta=meta_infer.infer_take)
registry.register("take_along_dim", "npu", take_along_dim, meta=meta_infer.infer_take_along_dim)
registry.register("masked_select", "npu", masked_select, meta=meta_infer.infer_masked_select)
registry.register("pad", "npu", pad, meta=meta_infer.infer_unary)
registry.register("pad_sequence", "npu", pad_sequence, meta=meta_infer.infer_pad_sequence)

# Critical tier operations
registry.register("mean", "npu", mean, meta=meta_infer.infer_sum)
registry.register("softmax", "npu", softmax, meta=meta_infer.infer_unary)
registry.register("log_softmax", "npu", log_softmax, meta=meta_infer.infer_unary)
registry.register("gelu", "npu", gelu, meta=meta_infer.infer_unary)
registry.register("layer_norm", "npu", layer_norm, meta=meta_infer.infer_unary)
registry.register("embedding", "npu", embedding, meta=meta_infer.infer_binary)

# Activation functions (composite ops)
registry.register("silu", "npu", silu, meta=meta_infer.infer_unary)
registry.register("leaky_relu", "npu", leaky_relu, meta=meta_infer.infer_unary)
registry.register("elu", "npu", elu, meta=meta_infer.infer_unary)
registry.register("mish", "npu", mish, meta=meta_infer.infer_unary)
registry.register("prelu", "npu", prelu, meta=meta_infer.infer_binary)

# Normalization (composite ops)
registry.register("batch_norm", "npu", batch_norm, meta=meta_infer.infer_unary)
registry.register("instance_norm", "npu", instance_norm, meta=meta_infer.infer_unary)
registry.register("group_norm", "npu", group_norm, meta=meta_infer.infer_unary)

# Tensor operations
# Random operations
registry.register("dropout", "npu", dropout, meta=meta_infer.infer_unary)

# Linalg operations
registry.register("linalg_qr", "npu", linalg_qr)

# Tensor indexing / selection ops
registry.register("narrow", "npu", narrow)
registry.register("select", "npu", select)
registry.register("expand", "npu", expand)
registry.register("masked_fill", "npu", masked_fill, meta=meta_infer.infer_unary)
registry.register("masked_fill_", "npu", masked_fill_, meta=meta_infer.infer_unary)
registry.register("index_put_", "npu", index_put_)
registry.register("index_put", "npu", index_put)
registry.register("index_copy_", "npu", index_copy_)
registry.register("index_fill_", "npu", index_fill_)
registry.register("index_add_", "npu", index_add_)
registry.register("scatter_", "npu", scatter_)
registry.register("scatter_add_", "npu", scatter_add_)
registry.register("masked_scatter_", "npu", masked_scatter_)
registry.register("unfold", "npu", unfold)

# Reduction ops (composite)
registry.register("var", "npu", var_, meta=meta_infer.infer_sum)
registry.register("norm", "npu", norm_, meta=meta_infer.infer_sum)
registry.register("prod", "npu", prod_, meta=meta_infer.infer_sum)
registry.register("floor_divide", "npu", floor_divide, meta=meta_infer.infer_binary)
registry.register("rms_norm", "npu", rms_norm, meta=meta_infer.infer_unary)

# Conv operations (ACLNN large kernels)
registry.register("conv2d", "npu", conv2d)
registry.register("conv1d", "npu", conv1d)
registry.register("conv_transpose2d", "npu", conv_transpose2d)
registry.register("conv_transpose1d", "npu", conv_transpose1d)

# Pooling operations (ACLNN large kernels)
registry.register("max_pool2d", "npu", max_pool2d)
registry.register("max_pool3d", "npu", max_pool3d)
registry.register("avg_pool2d", "npu", avg_pool2d)
registry.register("adaptive_avg_pool2d", "npu", adaptive_avg_pool2d)
registry.register("adaptive_max_pool2d", "npu", adaptive_max_pool2d)

# P1 ops
registry.register("std", "npu", std_, meta=meta_infer.infer_sum)
registry.register("reciprocal", "npu", reciprocal_, meta=meta_infer.infer_unary)
registry.register("addmm", "npu", addmm)
registry.register("einsum", "npu", einsum_)
registry.register("upsample_nearest2d", "npu", upsample_nearest2d)
registry.register("upsample_bilinear2d", "npu", upsample_bilinear2d)
registry.register("one_hot", "npu", one_hot)

# Logical ops
registry.register("logical_and", "npu", logical_and, meta=meta_infer.infer_binary_bool)
registry.register("logical_or", "npu", logical_or, meta=meta_infer.infer_binary_bool)
registry.register("logical_not", "npu", logical_not, meta=meta_infer.infer_unary_bool)

# In-place ops (batch 1)
registry.register("div_", "npu", div_, meta=meta_infer.infer_binary)

# P1 new ops
registry.register("baddbmm", "npu", baddbmm)
registry.register("trace", "npu", trace_op)
registry.register("cummin", "npu", cummin_op)
registry.register("logsumexp", "npu", logsumexp_op, meta=meta_infer.infer_sum)
registry.register("renorm", "npu", renorm_op, meta=meta_infer.infer_unary)
registry.register("logical_xor", "npu", logical_xor, meta=meta_infer.infer_binary_bool)

# P2 new ops
registry.register("nansum", "npu", nansum, meta=meta_infer.infer_sum)
registry.register("cross", "npu", cross_op, meta=meta_infer.infer_binary)

# P0: ACLNN large kernel ops
registry.register("im2col", "npu", im2col_op)
registry.register("grid_sample", "npu", grid_sample_op)
registry.register("affine_grid", "npu", affine_grid_op)

# P1: View/reshape ops
registry.register("broadcast_to", "npu", broadcast_to_op, meta=meta_infer.infer_broadcast_to)
registry.register("movedim", "npu", movedim_op, meta=meta_infer.infer_movedim)
registry.register("unflatten", "npu", unflatten_op, meta=meta_infer.infer_unflatten)
registry.register("diagonal", "npu", diagonal_op, meta=meta_infer.infer_diagonal)

# Missing forward ops — composites
registry.register("aminmax", "npu", aminmax_op)
registry.register("nanmean", "npu", nanmean_op, meta=meta_infer.infer_sum)
registry.register("argwhere", "npu", argwhere_op)
registry.register("det", "npu", det_op)
registry.register("diff", "npu", diff_op, meta=meta_infer.infer_unary)
registry.register("dist", "npu", dist_op)
registry.register("heaviside", "npu", heaviside_op, meta=meta_infer.infer_binary)
registry.register("inner", "npu", inner_op, meta=meta_infer.infer_binary)
registry.register("tensordot", "npu", tensordot_op)
registry.register("cdist", "npu", cdist_op)
registry.register("uniform", "npu", uniform_op)
registry.register("isreal", "npu", isreal_op, meta=meta_infer.infer_unary)
registry.register("isin", "npu", isin_op, meta=meta_infer.infer_binary)
registry.register("bincount", "npu", bincount_op)
registry.register("bucketize", "npu", bucketize_op)
registry.register("histc", "npu", histc_op)
registry.register("histogram", "npu", histogram_op)
registry.register("quantile", "npu", quantile_op)
registry.register("nanquantile", "npu", nanquantile_op)
registry.register("nanmedian", "npu", nanmedian_op)
registry.register("matrix_power", "npu", matrix_power_op, meta=meta_infer.infer_unary)
registry.register("col2im", "npu", col2im_op)

# Phase 1: ACLNN large kernel ops (910B confirmed working)
registry.register("mm", "npu", mm_op, meta=meta_infer.infer_matmul)
registry.register("bmm", "npu", bmm_op, meta=meta_infer.infer_matmul)
registry.register("special_digamma", "npu", special_digamma, meta=meta_infer.infer_unary)
registry.register("special_erfinv", "npu", special_erfinv, meta=meta_infer.infer_unary)
registry.register("special_gammaln", "npu", special_gammaln, meta=meta_infer.infer_unary)
registry.register("special_sinc", "npu", special_sinc, meta=meta_infer.infer_unary)
registry.register("linalg_inv", "npu", linalg_inv, meta=meta_infer.infer_unary)
registry.register("linalg_vector_norm", "npu", linalg_vector_norm_op)
registry.register("adaptive_avg_pool3d", "npu", adaptive_avg_pool3d_op)
registry.register("upsample_bicubic2d", "npu", upsample_bicubic2d_op)
registry.register("upsample_linear1d", "npu", upsample_linear1d_op)
registry.register("_adam_step", "npu", _adam_step_op)
registry.register("_adamw_step", "npu", _adamw_step_op)

# Upgrade composites to ACLNN large kernels
registry.register("aminmax", "npu", aminmax_aclnn)
registry.register("bincount", "npu", bincount_aclnn)

# Phase 2: Activation composites
registry.register("selu", "npu", selu_op, meta=meta_infer.infer_unary)
registry.register("celu", "npu", celu_op, meta=meta_infer.infer_unary)
registry.register("threshold", "npu", threshold_op, meta=meta_infer.infer_unary)
registry.register("hardshrink", "npu", hardshrink_op, meta=meta_infer.infer_unary)
registry.register("softshrink", "npu", softshrink_op, meta=meta_infer.infer_unary)
registry.register("hardswish", "npu", hardswish_op, meta=meta_infer.infer_unary)
registry.register("hardsigmoid", "npu", hardsigmoid_op, meta=meta_infer.infer_unary)
registry.register("softsign", "npu", softsign_op, meta=meta_infer.infer_unary)
registry.register("rrelu", "npu", rrelu_op, meta=meta_infer.infer_unary)
registry.register("normalize", "npu", normalize_op, meta=meta_infer.infer_unary)
registry.register("moveaxis", "npu", moveaxis_op, meta=meta_infer.infer_movedim)

# Phase 3: 1D pooling composites
registry.register("adaptive_avg_pool1d", "npu", adaptive_avg_pool1d_op)
registry.register("avg_pool1d", "npu", avg_pool1d_op)
registry.register("max_pool1d", "npu", max_pool1d_op)
registry.register("adaptive_max_pool1d", "npu", adaptive_max_pool1d_op)

# Phase 4: Optimizer composites
registry.register("_sgd_step", "npu", _sgd_step_op)
registry.register("_adagrad_step", "npu", _adagrad_step_op)
registry.register("_rmsprop_step", "npu", _rmsprop_step_op)
registry.register("_adadelta_step", "npu", _adadelta_step_op)
registry.register("_adamax_step", "npu", _adamax_step_op)
registry.register("_asgd_step", "npu", _asgd_step_op)
registry.register("_nadam_step", "npu", _nadam_step_op)
registry.register("_radam_step", "npu", _radam_step_op)
registry.register("_rprop_step", "npu", _rprop_step_op)
registry.register("_sparse_adam_step", "npu", _sparse_adam_step_op)

# Phase 5: Special function composites + CPU fallbacks
registry.register("special_entr", "npu", special_entr_op, meta=meta_infer.infer_unary)
registry.register("special_erfcx", "npu", special_erfcx_op, meta=meta_infer.infer_unary)
registry.register("special_logit", "npu", special_logit_op, meta=meta_infer.infer_unary)
registry.register("special_ndtr", "npu", special_ndtr_op, meta=meta_infer.infer_unary)
registry.register("special_log_ndtr", "npu", special_log_ndtr_op, meta=meta_infer.infer_unary)
registry.register("special_xlogy", "npu", special_xlogy_op, meta=meta_infer.infer_binary)
registry.register("special_xlog1py", "npu", special_xlog1py_op, meta=meta_infer.infer_binary)
registry.register("special_multigammaln", "npu", special_multigammaln_op)
registry.register("special_gammainc", "npu", special_gammainc_op)
registry.register("special_gammaincc", "npu", special_gammaincc_op)
registry.register("special_i0", "npu", special_i0_op, meta=meta_infer.infer_unary)
registry.register("special_i0e", "npu", special_i0e_op, meta=meta_infer.infer_unary)
registry.register("special_i1", "npu", special_i1_op, meta=meta_infer.infer_unary)
registry.register("special_i1e", "npu", special_i1e_op, meta=meta_infer.infer_unary)
registry.register("special_ndtri", "npu", special_ndtri_op, meta=meta_infer.infer_unary)
registry.register("special_polygamma", "npu", special_polygamma_op)
registry.register("special_zeta", "npu", special_zeta_op)

# Phase 6: Linalg composites + CPU fallbacks
registry.register("linalg_norm", "npu", linalg_norm_op)
registry.register("linalg_matrix_norm", "npu", linalg_matrix_norm_op)
registry.register("linalg_multi_dot", "npu", linalg_multi_dot_op)
registry.register("linalg_matrix_power", "npu", linalg_matrix_power_op)
registry.register("linalg_vander", "npu", linalg_vander_op)
registry.register("linalg_cholesky", "npu", linalg_cholesky_op)
registry.register("linalg_cond", "npu", linalg_cond_op)
registry.register("linalg_det", "npu", linalg_det_op)
registry.register("linalg_eig", "npu", linalg_eig_op)
registry.register("linalg_eigh", "npu", linalg_eigh_op)
registry.register("linalg_eigvals", "npu", linalg_eigvals_op)
registry.register("linalg_eigvalsh", "npu", linalg_eigvalsh_op)
registry.register("linalg_householder_product", "npu", linalg_householder_product_op)
registry.register("linalg_lstsq", "npu", linalg_lstsq_op)
registry.register("linalg_lu", "npu", linalg_lu_op)
registry.register("linalg_lu_factor", "npu", linalg_lu_factor_op)
registry.register("linalg_lu_solve", "npu", linalg_lu_solve_op)
registry.register("linalg_matrix_exp", "npu", linalg_matrix_exp_op)
registry.register("linalg_matrix_rank", "npu", linalg_matrix_rank_op)
registry.register("linalg_pinv", "npu", linalg_pinv_op)
registry.register("linalg_slogdet", "npu", linalg_slogdet_op)
registry.register("linalg_solve", "npu", linalg_solve_op)
registry.register("linalg_solve_triangular", "npu", linalg_solve_triangular_op)
registry.register("linalg_svd", "npu", linalg_svd_op)
registry.register("linalg_svdvals", "npu", linalg_svdvals_op)
registry.register("linalg_tensorinv", "npu", linalg_tensorinv_op)
registry.register("linalg_tensorsolve", "npu", linalg_tensorsolve_op)

# Phase 7: FFT CPU fallbacks
registry.register("fft_fft", "npu", fft_fft_op)
registry.register("fft_ifft", "npu", fft_ifft_op)
registry.register("fft_rfft", "npu", fft_rfft_op)
registry.register("fft_irfft", "npu", fft_irfft_op)
registry.register("fft_fft2", "npu", fft_fft2_op)
registry.register("fft_ifft2", "npu", fft_ifft2_op)
registry.register("fft_rfft2", "npu", fft_rfft2_op)
registry.register("fft_irfft2", "npu", fft_irfft2_op)
registry.register("fft_fftn", "npu", fft_fftn_op)
registry.register("fft_ifftn", "npu", fft_ifftn_op)
registry.register("fft_rfftn", "npu", fft_rfftn_op)
registry.register("fft_irfftn", "npu", fft_irfftn_op)
registry.register("fft_hfft", "npu", fft_hfft_op)
registry.register("fft_ihfft", "npu", fft_ihfft_op)
registry.register("fft_fftshift", "npu", fft_fftshift_op)
registry.register("fft_ifftshift", "npu", fft_ifftshift_op)

# Other missing ops: 3D conv, upsample, ctc_loss
registry.register("conv3d", "npu", conv3d_op)
registry.register("conv_transpose3d", "npu", conv_transpose3d_op)
registry.register("avg_pool3d", "npu", avg_pool3d_op)
registry.register("upsample_nearest1d", "npu", upsample_nearest1d_op)
registry.register("ctc_loss", "npu", ctc_loss_op)

__all__ = ["is_available", "_probe_model_dirs", "_model_dir", "allocator"]

