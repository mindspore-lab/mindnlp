from ..common import convert as convert_backend
from ..common import view as view_backend
from ..._dispatch.registry import registry
from .creation import empty_create_meta, ones_create_meta, tensor_create_meta, zeros_create_meta
from .ops import (
    _meta_binary_meta,
    _meta_binary_or_scalar_meta,
    _meta_matmul_meta,
    _meta_sum_meta,
    _meta_transpose_meta,
    _meta_unary_meta,
    _meta_unary_bool_meta,
    _meta_clamp_meta,
    _meta_clamp_min_meta,
    _meta_clamp_max_meta,
    _meta_hardtanh_meta,
    _meta_view_meta,
    _meta_contiguous_meta,
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
registry.register("fmin", "meta", _meta_binary_meta)
registry.register("fmax", "meta", _meta_binary_meta)
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
