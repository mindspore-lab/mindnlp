"""SoC-aware fallback policy routing and capability lookup for NPU ops."""

from . import runtime as npu_runtime
from . import ops_910a
from . import ops_910b
from . import ops_310p

_FALLBACK_OPS_310B = frozenset(
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
    }
)

_PROFILE_TO_POLICY = {
    "910a": ops_910a,
    "910b": ops_910b,
    "310p": ops_310p,
}

# Capability table keeps SoC-specific behavior switches centralized.
_PROFILE_CAPABILITIES = {
    "910a": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
        "fallback_ops": frozenset(),
    },
    "910b": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
        "fallback_ops": frozenset(),
    },
    "310b": {
        "use_smallop_arange_1d": True,
        "use_smallop_linspace": True,
        "fallback_ops": _FALLBACK_OPS_310B,
    },
    "310p": {
        "use_smallop_arange_1d": False,
        "use_smallop_linspace": False,
        "fallback_ops": frozenset(),
    },
}


def _resolve_profile(profile):
    return npu_runtime.soc_profile() if profile is None else str(profile).lower()


def capability(name, profile=None, default=False):
    caps = _PROFILE_CAPABILITIES.get(_resolve_profile(profile))
    if caps is None:
        return bool(default)
    return bool(caps.get(name, default))


def use_fallback(op_name, profile=None):
    profile_name = _resolve_profile(profile)
    ops = fallback_ops(profile=profile_name)
    if ops:
        return op_name in ops

    # Keep compatibility with legacy per-SoC policy modules where needed.
    policy = _PROFILE_TO_POLICY.get(profile_name)
    if policy is None:
        return False
    return policy.use_fallback(op_name)


def fallback_ops(profile=None):
    profile_name = _resolve_profile(profile)
    caps = _PROFILE_CAPABILITIES.get(profile_name)
    if caps is None:
        return frozenset()
    ops = caps.get("fallback_ops", frozenset())
    if isinstance(ops, frozenset):
        return ops
    return frozenset(ops)


def use_smallop_arange_1d(profile=None):
    return capability("use_smallop_arange_1d", profile=profile, default=False)


def use_smallop_linspace(profile=None):
    return capability("use_smallop_linspace", profile=profile, default=False)
