"""910B-specific NPU fallback policy table."""

_910B_FALLBACK_OPS = frozenset()


def use_fallback(op_name):
    return op_name in _910B_FALLBACK_OPS


def fallback_ops():
    return _910B_FALLBACK_OPS
