from typing import Callable, Any, Union, ParamSpec, TypeAlias
import torch
import mindspore as ms
import mindspore.numpy as mnp
import sys

P = ParamSpec('P')

TorchValue: TypeAlias = Union[torch.Tensor, torch.dtype, 'TorchCallable', Any]
TorchCallable: TypeAlias = Callable[P, TorchValue]
# Add JaxCallable for backward compatibility
JaxCallable: TypeAlias = Callable[P, Any]
# Use the correct type for MindSpore dtype - we can't use ms.dtype directly as it's a module
# Use type(ms.float32) to get the actual dtype type in MindSpore
DtypeType = type(ms.float32)
MSValue: TypeAlias = Union[ms.Tensor, DtypeType, 'MSCallable', Any]
MSCallable: TypeAlias = Callable[P, MSValue]

# For backward compatibility
MsValue = MSValue
MsCallable = MSCallable