"""Parametrizations module.

Provides ``torch.nn.utils.parametrizations`` namespace with weight_norm
and spectral_norm. Since the full parametrize framework is not yet
implemented, these delegate to the legacy implementations.
"""
from .weight_norm import weight_norm, remove_weight_norm
from .spectral_norm import spectral_norm

# Re-export under the parametrizations API names
__all__ = ['weight_norm', 'spectral_norm']
