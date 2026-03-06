"""Parametrization utilities (minimal compatibility shim).

Full parametrization support (torch.nn.utils.parametrize) is not yet
implemented.  This module provides the subset of the API that downstream
code queries at import time so that attribute checks and ``is_parametrized``
calls do not raise.
"""


def is_parametrized(module, tensor_name=None):
    """Check if a module has parametrizations.

    Always returns ``False`` because parametrizations are not yet
    implemented.
    """
    return False


def register_parametrization(module, tensor_name, parametrization, *, unsafe=False):
    """Register a parametrization on a tensor (stub -- not implemented)."""
    raise NotImplementedError(
        "torch.nn.utils.parametrize.register_parametrization is not yet implemented"
    )


def remove_parametrizations(module, tensor_name, leave_parametrized=True):
    """Remove parametrizations from a tensor (stub -- not implemented)."""
    raise NotImplementedError(
        "torch.nn.utils.parametrize.remove_parametrizations is not yet implemented"
    )


class _ParametrizationModule:  # pylint: disable=too-few-public-methods
    """Stub for parametrization support."""
