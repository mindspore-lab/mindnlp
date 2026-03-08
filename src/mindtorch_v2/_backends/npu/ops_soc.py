"""SoC-aware fallback policy routing and capability lookup for NPU ops."""

from . import runtime as npu_runtime
from . import ops_910a
from . import ops_910b
from . import ops_310b
from . import ops_310p

_PROFILE_TO_POLICY = {
    "910a": ops_910a,
    "910b": ops_910b,
    "310b": ops_310b,
    "310p": ops_310p,
}

# Capability table keeps SoC-specific behavior switches centralized.
_PROFILE_CAPABILITIES = {
    "910a": {
        "use_smallop_arange_1d": False,
    },
    "910b": {
        "use_smallop_arange_1d": False,
    },
    "310b": {
        "use_smallop_arange_1d": True,
    },
    "310p": {
        "use_smallop_arange_1d": False,
    },
}


def _resolve_profile(profile):
    return npu_runtime.soc_profile() if profile is None else str(profile).lower()


def use_fallback(op_name, profile=None):
    policy = _PROFILE_TO_POLICY.get(_resolve_profile(profile))
    if policy is None:
        return False
    return policy.use_fallback(op_name)


def fallback_ops(profile=None):
    policy = _PROFILE_TO_POLICY.get(_resolve_profile(profile))
    if policy is None:
        return frozenset()
    return policy.fallback_ops()


def capability(name, profile=None, default=False):
    caps = _PROFILE_CAPABILITIES.get(_resolve_profile(profile))
    if caps is None:
        return bool(default)
    return bool(caps.get(name, default))


def use_smallop_arange_1d(profile=None):
    return capability("use_smallop_arange_1d", profile=profile, default=False)
