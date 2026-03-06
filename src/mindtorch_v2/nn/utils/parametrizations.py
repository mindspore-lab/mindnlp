"""Parametrizations module.

Provides ``torch.nn.utils.parametrizations`` namespace with weight_norm
and spectral_norm. Since the full parametrize framework is not yet
implemented, these delegate to the legacy implementations.
"""
from .weight_norm import weight_norm, remove_weight_norm

# Re-export under the parametrizations API names
__all__ = ['weight_norm', 'spectral_norm']


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    """Apply spectral normalization to a parameter.

    Stub -- raises NotImplementedError.
    """
    raise NotImplementedError(
        "torch.nn.utils.parametrizations.spectral_norm is not yet implemented"
    )
