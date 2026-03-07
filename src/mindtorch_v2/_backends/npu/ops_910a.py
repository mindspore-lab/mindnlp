"""910A-specific NPU fallback policy table."""

_910A_FALLBACK_OPS = frozenset()


def use_fallback(op_name):
    return op_name in _910A_FALLBACK_OPS


def fallback_ops():
    return _910A_FALLBACK_OPS
