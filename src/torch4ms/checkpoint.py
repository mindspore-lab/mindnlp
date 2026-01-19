import torch
import os
from typing import Any, Dict, Optional
import numpy as np
from . import tensor
import mindspore as ms
import mindspore.numpy as mnp

# Use MindSpore for checkpoint functionality

def _to_mindspore(pytree):
  """Convert a pytree to MindSpore format.
  
  Args:
    pytree: The pytree to convert.
    
  Returns:
    The converted pytree with mindspore.Tensor leaves.
  """
  def to_mindspore_tensor(x):
    if isinstance(x, tensor.Tensor):
        return x.mindspore()
    elif isinstance(x, torch.Tensor):
        return ms.Tensor(x.cpu().numpy())
    return x
  # Use PyTorch's pytree utilities for tree traversal
  return torch.utils._pytree.tree_map(to_mindspore_tensor, pytree)


def _to_torch(pytree):
  """Convert a pytree to PyTorch format.
  
  Args:
    pytree: The pytree to convert.
    
  Returns:
    The converted pytree with torch.Tensor leaves.
  """
  def to_torch_tensor(x):
    if isinstance(x, ms.Tensor):
        return torch.from_numpy(np.asarray(x.asnumpy()))
    elif isinstance(x, tensor.Tensor):
        return x.torch()
    return x
  # Use PyTorch's pytree utilities for tree traversal
  return torch.utils._pytree.tree_map(to_torch_tensor, pytree)


def save_checkpoint(state: Dict[str, Any], path: str, step: int):
  """Saves a checkpoint to a file using MindSpore.

  Args:
    state: A dictionary containing the state to save. torch.Tensors will be
      converted to mindspore.Tensor.
    path: The path to save the checkpoint to.
    step: The training step.
  """
  # 确保目录存在
  os.makedirs(path, exist_ok=True)
  
  # 转换为MindSpore格式
  state = _to_mindspore(state)
  
  # 使用MindSpore的保存功能
  checkpoint_path = os.path.join(path, f"checkpoint_{step}.ckpt")
  ms.save_checkpoint(state, checkpoint_path)


def load_checkpoint(path: str) -> Dict[str, Any]:
  """Loads a checkpoint and returns it using MindSpore.

  This function can load both PyTorch-style (single file) and MindSpore-style
  checkpoints.

  Args:
    path: The path to the checkpoint file or directory.

  Returns:
    The loaded state with mindspore.Tensor leaves.
  """
  if os.path.isdir(path):
    # MindSpore-style checkpoint directory
    # 查找目录中的最新检查点文件
    checkpoint_files = [f for f in os.listdir(path) if f.startswith("checkpoint_") and f.endswith(".ckpt")]
    if not checkpoint_files:
      raise FileNotFoundError(f"No checkpoint files found in directory {path}")
    
    # 按步骤号排序，选择最新的
    checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    checkpoint_path = os.path.join(path, checkpoint_files[0])
    
    # 使用MindSpore加载
    state = ms.load_checkpoint(checkpoint_path)
    return state
  elif os.path.isfile(path):
    # 直接加载文件
    try:
      # 尝试作为MindSpore检查点加载
      state = ms.load_checkpoint(path)
      return state
    except:
      # 如果失败，尝试作为PyTorch检查点加载并转换
      state = torch.load(path, weights_only=False)
      return _to_mindspore(state)
  else:
    raise FileNotFoundError(f"No such file or directory: {path}")
    