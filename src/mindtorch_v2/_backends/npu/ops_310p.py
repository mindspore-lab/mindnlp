"""310P-specific NPU fallback policy table."""

_310P_FALLBACK_OPS = frozenset()


def use_fallback(op_name):
    return op_name in _310P_FALLBACK_OPS


def fallback_ops():
    return _310P_FALLBACK_OPS
