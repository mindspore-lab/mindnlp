import functools
import numpy as np
import torch
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as mnp
from torch4ms.ops import mappings
from torch4ms.view import View
from torch4ms import types
import sys

from typing import Callable, Optional, ParamSpec, Concatenate


class InplaceOp:

  def __init__(self,
               functional_op,
               replace=False,
               position_to_mutate=0,
               is_mindspore_func=False):
    self.functional = functional_op
    self.replace = replace
    self.position_to_mutate = position_to_mutate
    self.is_mindspore_func = is_mindspore_func

  def __call__(self, *args, **kwargs):
    to_mutate = args[self.position_to_mutate]
    view_value = to_mutate
    if isinstance(to_mutate, View):
      view_value = to_mutate.torch()
    env = view_value._env

    if self.is_mindspore_func:
      view_value, args, kwargs = env.t2ms_iso((view_value, args, kwargs))
      new_value_ms = self.functional(view_value, *args[1:], **kwargs)
      new_value = env.ms2t_iso(new_value_ms)
    else:
      new_value = self.functional(view_value, *args[1:], **kwargs)

    if isinstance(to_mutate, View):
      to_mutate.update(new_value)
    else:
      if self.replace:
        to_mutate._elem = new_value._elem
      else:
        to_mutate.copy_(new_value)
    return to_mutate


class OutVariant:

  def __call__(self, *args, **kwargs):
    to_mutate = kwargs.pop('out')
    to_mutate._elem = self.functional(*args, **kwargs)._elem
    return to_mutate


P = ParamSpec('P')


def convert_dtype(use_default_dtype: bool = True):
  """Converts `dtype` kwarg of function from torch to MindSpore."""

  def decorator(func: types.TorchCallable):

    @functools.wraps(func)
    def wrapper(*args: P.args,
                dtype: Optional[torch.dtype] = None,
                **kwargs: P.kwargs):
      if not dtype and use_default_dtype:
        dtype = torch.get_default_dtype()
      if isinstance(dtype, torch.dtype):
        ms_dtype = mappings.t2ms_dtype(dtype)
      else:
        ms_dtype = dtype

      return func(*args, dtype=ms_dtype, **kwargs)

    return wrapper

  return decorator


def maybe_convert_constant_dtype(val: Optional[types.MSValue],
                                 dtype: Optional[ms.Type]):
  """Optionally converts scalar constant's dtype using `numpy`"""
  if val and dtype:
    if isinstance(val, ms.Tensor):
      return maybe_convert_constant_dtype(val.asnumpy().item(), dtype)
    return np.array(val, dtype)
  return val


def promote_int_input(f: Callable[Concatenate[ms.Tensor, P], types.MSValue]):
  """If the first argument is an int tensor, promote it to float32."""

  @functools.wraps(f)
  def wrapper(x: ms.Tensor, *args: P.args, **kwargs: P.kwargs):
    if x.dtype in (ms.int8, ms.int16, ms.int32, ms.int64):
      x = x.astype(mappings.t2ms_dtype(torch.get_default_dtype()))
    return f(x, *args, **kwargs)

  return wrapper


def foreach_loop(seq: ms.Tensor,
                 fn: Callable[[ms.Tensor, ms.Tensor], ms.Tensor],
                 init_val=0.0):
  """Run `fn` for each element of 1-D tensor `seq`."""
  if seq.ndim != 1:
    raise ValueError("seq must be 1-D")
  # MindSpore 无 fori_loop，用 while + mutable 实现
  carry = ms.mutable(init_val)
  i = ms.mutable(0)
  while i < seq.shape[0]:
    carry = fn(carry, seq[i])
    i += 1
  return carry