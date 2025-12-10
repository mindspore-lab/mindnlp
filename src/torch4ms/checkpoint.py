import torch
import os
from typing import Any, Dict, Optional
import numpy as np
from . import tensor

# Try to import Flax and JAX, but handle case when they're not available
FLAX_AVAILABLE = False
JAX_AVAILABLE = False
try:
    from flax.training import checkpoints
    FLAX_AVAILABLE = True
except ImportError:
    pass

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    pass

def _to_jax(pytree):
  """Convert a pytree to JAX format.
  
  Args:
    pytree: The pytree to convert.
    
  Returns:
    The converted pytree with jax.Array leaves.
    
  Raises:
    ImportError: If JAX is not available.
  """
  if not JAX_AVAILABLE:
    raise ImportError("JAX is not available. This function requires JAX.")
    
  def to_jax_array(x):
    if isinstance(x, tensor.Tensor):
        return x.jax()
    elif isinstance(x, torch.Tensor):
        return jnp.asarray(x.cpu().numpy())
    return x
  return jax.tree_util.tree_map(to_jax_array, pytree)


def _to_torch(pytree):
  """Convert a pytree to PyTorch format.
  
  Args:
    pytree: The pytree to convert.
    
  Returns:
    The converted pytree with torch.Tensor leaves.
    
  Raises:
    ImportError: If JAX is not available.
  """
  if not JAX_AVAILABLE:
    raise ImportError("JAX is not available. This function requires JAX.")
    
  return jax.tree_util.tree_map(
    lambda x: torch.from_numpy(np.asarray(x))
    if isinstance(x, (jnp.ndarray, jax.Array)) else x, pytree)


def save_checkpoint(state: Dict[str, Any], path: str, step: int):
  """Saves a checkpoint to a file in JAX style.

  Args:
    state: A dictionary containing the state to save. torch.Tensors will be
      converted to jax.Array.
    path: The path to save the checkpoint to. This is a directory.
    step: The training step.
    
  Raises:
    ImportError: If Flax or JAX is not available.
  """
  if not FLAX_AVAILABLE:
    raise ImportError("Flax is not available. This function requires Flax.")
  
  state = _to_jax(state)
  checkpoints.save_checkpoint(path, state, step=step, overwrite=True)


def load_checkpoint(path: str) -> Dict[str, Any]:
  """Loads a checkpoint and returns it in JAX format.

  This function can load both PyTorch-style (single file) and JAX-style
  (directory) checkpoints.

  If the checkpoint is in PyTorch format, it will be converted to JAX format.

  Args:
    path: The path to the checkpoint.

  Returns:
    The loaded state in JAX format (pytree with jax.Array leaves).
    
  Raises:
    ImportError: If Flax or JAX is not available.
  """
  if os.path.isdir(path):
    # JAX-style checkpoint
    if not FLAX_AVAILABLE:
      raise ImportError("Flax is not available. Cannot load JAX-style checkpoint.")
    state = checkpoints.restore_checkpoint(path, target=None)
    if state is None:
      raise FileNotFoundError(f"No checkpoint found at {path}")
    return state
  elif os.path.isfile(path):
    # PyTorch-style checkpoint
    state = torch.load(path, weights_only=False)
    return _to_jax(state)
  else:
    raise FileNotFoundError(f"No such file or directory: {path}")
    