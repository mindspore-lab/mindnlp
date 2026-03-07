"""SoC-aware fallback policy routing for NPU ops."""

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
