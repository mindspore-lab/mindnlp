"""typing"""
from typing import Callable, Tuple, Type, Union

from mindnlp.core.nn import Module


LayerType = Union[str, Callable, Type[Module]]
PadType = Union[str, int, Tuple[int, int]]
