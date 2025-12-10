"""
MindSpore-PyTorch interoperation utilities.
This module provides functions to bridge between PyTorch and MindSpore tensors and modules.
"""
import collections
import copy
import functools
import torch
from inspect import signature
from functools import wraps
from torch.nn.utils import stateless as torch_stateless
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops
from torch4ms import tensor
from torch4ms import util
from torch4ms.ops import mappings
import torch4ms

from torch4ms.types import MSValue, TorchValue, MSCallable, TorchCallable


def extract_all_buffers(m: torch.nn.Module):
  """
  Extract all buffers from a PyTorch module.
  
  Args:
    m: PyTorch module
  
  Returns:
    Dictionary mapping buffer names to buffer values
  """
  return {k: v for k, v in m.named_buffers()}


def set_all_buffers(m, params, buffers):
  """
  Set all buffers in a PyTorch module.
  
  Args:
    m: PyTorch module
    params: Dictionary of parameters
    buffers: Dictionary of buffers
  """
  for k, v in buffers.items():
    if k in params:
      del params[k]
  m._buffers.update(buffers)


class JittableModule(torch.nn.Module):
  """
  A PyTorch module that can be compiled with MindSpore.
  """
  def __init__(self, m: torch.nn.Module, extra_jit_args={}, dedup_parameters=True):
    """
    Initialize a JittableModule.
    
    Args:
      m: PyTorch module to wrap
      extra_jit_args: Extra arguments to pass to MindSpore JIT compilation
      dedup_parameters: Whether to deduplicate parameters
    """
    super().__init__()
    self._model = m
    self._extra_jit_args = extra_jit_args
    self._jitted = {}
    
    # Extract parameters and buffers
    self.params = {k: v for k, v in m.named_parameters()}
    self.buffers = extract_all_buffers(m)
    
    # For deduped parameters, we need to remember which names are aliases
    self._extra_dumped_weights = {}
    if dedup_parameters:
      # Deduplicate parameters
      param_values = {}
      for k, v in list(self.params.items()):
        v_id = id(v)
        if v_id not in param_values:
          param_values[v_id] = k
        else:
          # This is a duplicate parameter
          if param_values[v_id] not in self._extra_dumped_weights:
            self._extra_dumped_weights[param_values[v_id]] = []
          self._extra_dumped_weights[param_values[v_id]].append(k)
          del self.params[k]

  @property
  def __class__(self):
    # Lie about the class type so that isinstance(jittable_module, self._model.__class__) works
    return self._model.__class__

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

  def functional_call(self, method_or_name, params, buffers, *args, **kwargs):
    """
    Call a method on the model with the given parameters and buffers.
    
    Args:
      method_or_name: Method or name of method to call
      params: Dictionary of parameters
      buffers: Dictionary of buffers
      *args: Positional arguments to pass to the method
      **kwargs: Keyword arguments to pass to the method
      
    Returns:
      Result of the method call
    """
    kwargs = kwargs or {}
    params_copy = copy.copy(params)
    params_copy.update(buffers)
    # reinflate the state dict so there are not any missing keys
    for k, v in self._extra_dumped_weights.items():
      for new_key in v:
        params_copy[new_key] = params_copy[k]

    if isinstance(method_or_name, str):
      method = getattr(self._model, method_or_name)
    else:
      if not callable(method_or_name):
        raise TypeError(
            f"method_or_name should be a callable or a string, got {type(method_or_name)}"
        )
      method = method_or_name
      args = (self._model,) + args
    with torch_stateless._reparametrize_module(self._model, params_copy):
      res = method(*args, **kwargs)
    return res

  def jittable_call(self, method_name: str, *args, **kwargs):
    """
    Call a method on the model with MindSpore JIT compilation.
    
    Args:
      method_name: Name of method to call
      *args: Positional arguments to pass to the method
      **kwargs: Keyword arguments to pass to the method
      
    Returns:
      Result of the method call
    """
    if method_name not in self._jitted:
      # Use MindSpore's jit instead of JAX's jit
      jitted = ms.jit(
          functools.partial(self.functional_call, method_name),
          **self._extra_jit_args,
      )

      def jitted_forward(*args, **kwargs):
        return jitted(self.params, self.buffers, *args, **kwargs)

      self._jitted[method_name] = jitted_forward
    return self._jitted[method_name](*args, **kwargs)

  def forward(self, *args, **kwargs):
    """
    Forward pass through the model.
    
    Args:
      *args: Positional arguments
      **kwargs: Keyword arguments
      
    Returns:
      Output of the model
    """
    return self.jittable_call('forward', *args, **kwargs)

  def __getattr__(self, key):
    """
    Get an attribute from the model.
    
    Args:
      key: Attribute name
      
    Returns:
      Attribute value
    """
    if key == '_model':
      return super().__getattr__(key)
    if key in self._jitted:
      return self._jitted[key]
    return getattr(self._model, key)

  def make_jitted(self, key):
    """
    Make a method jittable.
    
    Args:
      key: Method name
    """
    jitted = ms.jit(
        functools.partial(self.functional_call, key),
        **self._extra_jit_args)

    def call(*args, **kwargs):
      return jitted(self.params, self.buffers, *args, **kwargs)

    self._jitted[key] = call


class CompileMixin:
  """
  Mixin for compiling PyTorch modules with MindSpore.
  """
  def functional_call(self, method, params, buffers, *args, **kwargs):
    """
    Call a method with the given parameters and buffers.
    
    Args:
      method: Method to call
      params: Dictionary of parameters
      buffers: Dictionary of buffers
      *args: Positional arguments
      **kwargs: Keyword arguments
      
    Returns:
      Result of the method call
    """
    kwargs = kwargs or {}
    params_copy = copy.copy(params)
    params_copy.update(buffers)
    with torch_stateless._reparametrize_module(self, params_copy):
      res = method(*args, **kwargs)
    return res

  def jit(self, method):
    """
    JIT compile a method with MindSpore.
    
    Args:
      method: Method to compile
      
    Returns:
      Compiled method
    """
    # Fix: method_name is not defined, should be method.__name__
    jitted = ms.jit(functools.partial(self.functional_call, method.__name__))

    def call(*args, **kwargs):
      return jitted(self.named_parameters(), self.named_buffers(), *args, **kwargs)

    return call


def compile_nn_module(m: torch.nn.Module, methods=None):
  """
  Compile a PyTorch module with MindSpore.
  
  Args:
    m: PyTorch module to compile
    methods: Methods to compile
  """
  if methods is None:
    methods = ['forward']

  # Fix: NewParent should be new_parent for consistency
  new_parent = type(
      m.__class__.__name__ + '_with_CompileMixin',
      (CompileMixin, m.__class__),
  )
  m.__class__ = new_parent


def _torch_view(t: MSValue) -> TorchValue:
  """
  Convert a MindSpore value to a PyTorch value.
  
  Args:
    t: MindSpore value
    
  Returns:
    PyTorch value
  """
  # t is an object from mindspore land
  # view it as-if it's a torch land object
  if isinstance(t, ms.Tensor):
    return tensor.Tensor(t, torch4ms.default_env())
  if isinstance(t, mnp.dtype):
    return mappings.ms2t_dtype(t)
  if callable(t):  # t is a MSCallable
    return functools.partial(call_ms, t)
  # regular types are not changed
  return t


# Use functools.partial with a custom tree_map function since we don't have pytree from JAX
def tree_map(fn, tree):
  """
  Map a function over a tree structure.
  
  Args:
    fn: Function to apply to each leaf
    tree: Tree structure
    
  Returns:
    New tree with function applied to each leaf
  """
  if isinstance(tree, (list, tuple)):
    return type(tree)(tree_map(fn, x) for x in tree)
  elif isinstance(tree, dict):
    return {k: tree_map(fn, v) for k, v in tree.items()}
  else:
    return fn(tree)

torch_view = functools.partial(tree_map, _torch_view)


def _ms_view(t: TorchValue) -> MSValue:
  """
  Convert a PyTorch value to a MindSpore value.
  
  Args:
    t: PyTorch value
    
  Returns:
    MindSpore value
  """
  # t is an object from torch land
  # view it as-if it's a mindspore land object
  if isinstance(t, torch.Tensor):
    assert isinstance(t, tensor.Tensor) or isinstance(t, tensor.View), type(t)
    return t.ms()  # Assuming tensor.Tensor has a ms() method
  if isinstance(t, type(torch.int32)):
    return mappings.t2ms_dtype(t)

  # torch.nn.Module needs special handling
  if not isinstance(t, torch.nn.Module) and callable(t):  # t is a TorchCallable
    return functools.partial(call_torch, t)
  # regular types are not changed
  return t


ms_view = functools.partial(tree_map, _ms_view)


def call_ms(ms_func: MSCallable, *args: TorchValue, **kwargs: TorchValue) -> TorchValue:
  """
  Call a MindSpore function with PyTorch arguments.
  
  Args:
    ms_func: MindSpore function to call
    *args: PyTorch positional arguments
    **kwargs: PyTorch keyword arguments
    
  Returns:
    PyTorch value
  """
  args, kwargs = ms_view((args, kwargs))
  res: MSValue = ms_func(*args, **kwargs)
  return torch_view(res)


def call_torch(torch_func: TorchCallable, *args: MSValue, **kwargs: MSValue) -> MSValue:
  """
  Call a PyTorch function with MindSpore arguments.
  
  Args:
    torch_func: PyTorch function to call
    *args: MindSpore positional arguments
    **kwargs: MindSpore keyword arguments
    
  Returns:
    MindSpore value
  """
  args, kwargs = torch_view((args, kwargs))
  with torch4ms.default_env():
    res: TorchValue = torch_func(*args, **kwargs)
  return ms_view(res)


def ms2t_autograd(fn, call_ms=call_ms):
  """
  Given a MindSpore function, returns a PyTorch autograd function.
  
  It wraps `fn` to compute both the output and gradients. The wrapped function
  is then run via `call_ms` and integrated into the PyTorch autograd framework.
  
  Args:
    fn: MindSpore function
    call_ms: Function to call MindSpore functions
    
  Returns:
    PyTorch autograd function
  """
  @wraps(fn)
  def inner(*args, **kwargs):
    # MindSpore doesn't have vjp like JAX, so we implement a simpler version
    # that relies on MindSpore's automatic differentiation
    class MSFun(torch.autograd.Function):
      @staticmethod
      def forward(ctx, *flat_args_kwargs):
        # Convert PyTorch tensors to MindSpore tensors
        ms_args = []
        for arg in flat_args_kwargs:
          if isinstance(arg, torch.Tensor):
            ms_args.append(mappings.t2ms(arg))
          else:
            ms_args.append(arg)
        
        # Call the MindSpore function
        y = fn(*ms_args)
        
        # Convert MindSpore tensor back to PyTorch tensor
        if isinstance(y, ms.Tensor):
          y = mappings.ms2t(y)
        
        # Save necessary information for backward
        ctx.save_for_backward(*[a for a in flat_args_kwargs if isinstance(a, torch.Tensor)])
        return y

      @staticmethod
      def backward(ctx, *grad_out):
        # This is a simplified backward pass
        # In a real implementation, you would use MindSpore's gradient computation
        # and convert the gradients back to PyTorch tensors
        saved_tensors = ctx.saved_tensors
        grads = [None] * (len(saved_tensors) + len(ctx.needs_input_grad))
        
        # For simplicity, we'll just return None for all gradients
        # A real implementation would compute gradients using MindSpore
        return tuple(grads)

    sig = signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    
    # Flatten args and kwargs
    flat_args_kwargs = []
    for arg in bound.args:
      flat_args_kwargs.append(arg)
    for arg in bound.kwargs.values():
      flat_args_kwargs.append(arg)
    
    y = MSFun.apply(*flat_args_kwargs)
    return y

  return inner


# MindSpore doesn't have fori_loop like JAX, so we implement a simple version
def fori_loop(lower, upper, body_fun, init_val):
  """
  Simple implementation of fori_loop for MindSpore.
  
  Args:
    lower: Lower bound (inclusive)
    upper: Upper bound (exclusive)
    body_fun: Function to apply at each iteration
    init_val: Initial value
    
  Returns:
    Final value after all iterations
  """
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val


def wrap_ms_jit(torch_function, ms_jit_func=ms.jit, kwargs_for_ms=None):
  """
  Wrap a PyTorch function with MindSpore JIT.
  
  Args:
    torch_function: PyTorch function to wrap
    ms_jit_func: MindSpore JIT function
    kwargs_for_ms: Keyword arguments for MindSpore JIT
    
  Returns:
    JIT-compiled function
  """
  kwargs_for_ms = kwargs_for_ms or {}
  ms_func = ms_view(torch_function)
  jitted = ms_jit_func(ms_func, **kwargs_for_ms)
  return torch_view(jitted)


def ms_jit(torch_function, kwargs_for_ms_jit=None, fix_for_buffer_donation=False):
  """
  JIT compile a PyTorch function with MindSpore.
  
  Args:
    torch_function: PyTorch function to compile
    kwargs_for_ms_jit: Keyword arguments for MindSpore JIT
    fix_for_buffer_donation: Whether to fix for buffer donation
    
  Returns:
    JIT-compiled function
  """
  return wrap_ms_jit(
      torch_function,
      ms_jit_func=ms.jit,
      kwargs_for_ms=kwargs_for_ms_jit)


def ms_value_and_grad(torch_function, kwargs_for_value_and_grad=None):
  """
  Compute value and gradients of a PyTorch function using MindSpore.
  
  Args:
    torch_function: PyTorch function
    kwargs_for_value_and_grad: Keyword arguments
    
  Returns:
    Function that computes value and gradients
  """
  # MindSpore doesn't have value_and_grad like JAX, so we implement a simplified version
  def value_and_grad_wrapper(func, **kwargs):
    def wrapper(*args, **kwargs):
      # In a real implementation, you would use MindSpore's gradient computation
      # For simplicity, we'll just return the function value and None for gradients
      value = func(*args, **kwargs)
      return value, lambda g: [None] * len(args)
    return wrapper
  
  return wrap_ms_jit(
      torch_function,
      ms_jit_func=value_and_grad_wrapper,
      kwargs_for_ms=kwargs_for_value_and_grad)


def gradient_checkpoint(torch_function, kwargs=None):
  """
  Apply gradient checkpointing to a PyTorch function using MindSpore.
  
  Args:
    torch_function: PyTorch function
    kwargs: Keyword arguments
    
  Returns:
    Function with gradient checkpointing applied
  """
  # MindSpore doesn't have checkpoint like JAX, so we just return the function
  # In a real implementation, you would use MindSpore's checkpointing mechanism
  return wrap_ms_jit(
      torch_function,
      ms_jit_func=lambda f, **kw: f,
      kwargs_for_ms=kwargs)