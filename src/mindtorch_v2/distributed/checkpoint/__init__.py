"""Distributed checkpoint APIs available in mindtorch_v2."""

from .state_dict import get_state_dict, set_state_dict

__all__ = [
    "get_state_dict",
    "set_state_dict",
]
