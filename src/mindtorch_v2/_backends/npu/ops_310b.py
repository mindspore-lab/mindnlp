"""310B-specific NPU fallback policy table (compat shim)."""

# Source of truth moved to ops_soc capability table. Keep this module for
# compatibility with any out-of-tree imports.
_310B_FALLBACK_OPS = frozenset()


def use_fallback(op_name):
    return op_name in _310B_FALLBACK_OPS


def fallback_ops():
    return _310B_FALLBACK_OPS
