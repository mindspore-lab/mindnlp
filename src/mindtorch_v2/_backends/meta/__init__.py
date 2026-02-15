from ..common import convert as convert_backend
from ..common import view as view_backend
from ..._dispatch.registry import registry
from .creation import empty_create_meta, ones_create_meta, tensor_create_meta, zeros_create_meta
from .ops import (
    _meta_binary_meta,
    _meta_matmul_meta,
    _meta_sum_meta,
    _meta_transpose_meta,
    _meta_unary_meta,
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
