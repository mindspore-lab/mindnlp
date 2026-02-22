from ..common import convert as convert_backend
from ..common import view as view_backend
from ..._dispatch.registry import registry
from .creation import (
    arange_create_meta,
    empty_create_meta,
    full_create_meta,
    eye_create_meta,
    linspace_create_meta,
    logspace_create_meta,
    ones_create_meta,
    range_create_meta,
    tensor_create_meta,
    zeros_create_meta,
)
from .ops import (
    _meta_binary_meta,
    _meta_binary_or_scalar_meta,
    _meta_matmul_meta,
    _meta_sum_meta,
    _meta_reduce_bool_meta,
    _meta_argmax_meta,
    _meta_transpose_meta,
    _meta_unary_meta,
    _meta_unary_bool_meta,
    _meta_clamp_meta,
    _meta_clamp_min_meta,
    _meta_clamp_max_meta,
    _meta_hardtanh_meta,
    _meta_where_meta,
    _meta_lerp_meta,
    _meta_addcmul_meta,
    _meta_addcdiv_meta,
    _meta_view_meta,
    _meta_contiguous_meta,
    _meta_equal_meta,
    _meta_cummax_meta,
    _meta_argsort_meta,
    _meta_sort_meta,
    _meta_topk_meta,
)

registry.register("add", "meta", _meta_binary_meta)
registry.register("mul", "meta", _meta_binary_meta)
registry.register("matmul", "meta", _meta_matmul_meta)
registry.register("relu", "meta", _meta_unary_meta)
registry.register("abs", "meta", _meta_unary_meta)
registry.register("neg", "meta", _meta_unary_meta)
registry.register("exp", "meta", _meta_unary_meta)
registry.register("log", "meta", _meta_unary_meta)
registry.register("sqrt", "meta", _meta_unary_meta)
registry.register("sin", "meta", _meta_unary_meta)
registry.register("cos", "meta", _meta_unary_meta)
registry.register("tan", "meta", _meta_unary_meta)
registry.register("tanh", "meta", _meta_unary_meta)
registry.register("sigmoid", "meta", _meta_unary_meta)
registry.register("floor", "meta", _meta_unary_meta)
registry.register("ceil", "meta", _meta_unary_meta)
registry.register("round", "meta", _meta_unary_meta)
registry.register("trunc", "meta", _meta_unary_meta)
registry.register("frac", "meta", _meta_unary_meta)
registry.register("pow", "meta", _meta_binary_or_scalar_meta)
registry.register("log2", "meta", _meta_unary_meta)
registry.register("log10", "meta", _meta_unary_meta)
registry.register("exp2", "meta", _meta_unary_meta)
registry.register("rsqrt", "meta", _meta_unary_meta)
registry.register("sign", "meta", _meta_unary_meta)
registry.register("signbit", "meta", _meta_unary_bool_meta)
registry.register("isnan", "meta", _meta_unary_bool_meta)
registry.register("isinf", "meta", _meta_unary_bool_meta)
registry.register("isfinite", "meta", _meta_unary_bool_meta)
registry.register("sinh", "meta", _meta_unary_meta)
registry.register("cosh", "meta", _meta_unary_meta)
registry.register("asinh", "meta", _meta_unary_meta)
registry.register("acosh", "meta", _meta_unary_meta)
registry.register("atanh", "meta", _meta_unary_meta)
registry.register("erf", "meta", _meta_unary_meta)
registry.register("erfc", "meta", _meta_unary_meta)
registry.register("softplus", "meta", _meta_unary_meta)
registry.register("clamp", "meta", _meta_clamp_meta)
registry.register("clamp_min", "meta", _meta_clamp_min_meta)
registry.register("clamp_max", "meta", _meta_clamp_max_meta)
registry.register("relu6", "meta", _meta_unary_meta)
registry.register("hardtanh", "meta", _meta_hardtanh_meta)
registry.register("min", "meta", _meta_binary_meta)
registry.register("max", "meta", _meta_binary_meta)
registry.register("amin", "meta", _meta_sum_meta)
registry.register("amax", "meta", _meta_sum_meta)
registry.register("all", "meta", _meta_reduce_bool_meta)
registry.register("any", "meta", _meta_reduce_bool_meta)
registry.register("argmax", "meta", _meta_argmax_meta)
registry.register("argmin", "meta", _meta_argmax_meta)
registry.register("count_nonzero", "meta", _meta_argmax_meta)
registry.register("cumsum", "meta", _meta_unary_meta)
registry.register("cumprod", "meta", _meta_unary_meta)
registry.register("cummax", "meta", _meta_cummax_meta)
registry.register("argsort", "meta", _meta_argsort_meta)
registry.register("sort", "meta", _meta_sort_meta)
registry.register("topk", "meta", _meta_topk_meta)
registry.register("allclose", "meta", _meta_reduce_bool_meta)
registry.register("isclose", "meta", _meta_binary_meta)
registry.register("equal", "meta", _meta_equal_meta)
registry.register("fmin", "meta", _meta_binary_meta)
registry.register("fmax", "meta", _meta_binary_meta)
registry.register("where", "meta", _meta_where_meta)
registry.register("atan", "meta", _meta_unary_meta)
registry.register("atan2", "meta", _meta_binary_meta)
registry.register("asin", "meta", _meta_unary_meta)
registry.register("acos", "meta", _meta_unary_meta)
registry.register("lerp", "meta", _meta_lerp_meta)
registry.register("addcmul", "meta", _meta_addcmul_meta)
registry.register("addcdiv", "meta", _meta_addcdiv_meta)
registry.register("logaddexp", "meta", _meta_binary_meta)
registry.register("logaddexp2", "meta", _meta_binary_meta)
registry.register("hypot", "meta", _meta_binary_meta)
registry.register("remainder", "meta", _meta_binary_meta)
registry.register("fmod", "meta", _meta_binary_meta)
registry.register("contiguous", "meta", _meta_contiguous_meta)
registry.register("sum", "meta", _meta_sum_meta)
registry.register("reshape", "meta", view_backend.reshape, meta=_meta_view_meta)
registry.register("view", "meta", view_backend.view, meta=_meta_view_meta)
registry.register("transpose", "meta", view_backend.transpose, meta=_meta_transpose_meta)
registry.register("to", "meta", convert_backend.to_device)

registry.register("tensor", "meta", tensor_create_meta)
registry.register("zeros", "meta", zeros_create_meta)
registry.register("ones", "meta", ones_create_meta)
registry.register("empty", "meta", empty_create_meta)
registry.register("arange", "meta", arange_create_meta)
registry.register("linspace", "meta", linspace_create_meta)
registry.register("full", "meta", full_create_meta)
registry.register("logspace", "meta", logspace_create_meta)
registry.register("eye", "meta", eye_create_meta)
registry.register("range", "meta", range_create_meta)
