"""310B-specific NPU fallback policy table."""

_310B_FALLBACK_OPS = frozenset(
    {
        "atan2",
        "where",
        "flip",
        "argsort",
        "sort",
        "topk",
        "diag",
        "lerp",
        "remainder",
        "isclose",
        "softplus",
        "uniform_",
        "normal_",
        "layer_norm",
        "mish",
        "batch_norm",
        "dropout",
        "take_along_dim",
        "gather",
        "linspace",
    }
)


def use_fallback(op_name):
    return op_name in _310B_FALLBACK_OPS


def fallback_ops():
    return _310B_FALLBACK_OPS
