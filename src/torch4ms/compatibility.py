# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
兼容性层模块，提供PyTorch API到MindSpore API的转换功能
"""

import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    import mindspore as ms
    import mindspore.ops as ops

    MS_AVAILABLE = True
except ImportError:
    MS_AVAILABLE = False
    logger.warning("MindSpore not available, compatibility layer will use stub functions")

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, compatibility layer will use stub functions")

# 数据类型映射
DTYPE_MAPPING: Dict[Any, Any] = {}
if MS_AVAILABLE and TORCH_AVAILABLE:
    # PyTorch -> MindSpore 数据类型映射
    DTYPE_MAPPING = {
        torch.float32: ms.float32,
        torch.float64: ms.float64,
        torch.int32: ms.int32,
        torch.int64: ms.int64,
        torch.bool: ms.bool_,
        torch.uint8: ms.uint8,
        torch.int8: ms.int8,
        torch.int16: ms.int16,
    }

# 设备类型映射
DEVICE_MAPPING: Dict[str, str] = {
    'cuda': 'GPU',
    'cpu': 'CPU',
    'npu': 'NPU',
}


def torch_dtype_to_ms_dtype(torch_dtype: Any) -> Any:
    """
    将PyTorch数据类型转换为MindSpore数据类型

    Args:
        torch_dtype: PyTorch数据类型

    Returns:
        MindSpore数据类型
    """
    if not MS_AVAILABLE or not TORCH_AVAILABLE:
        return torch_dtype

    if torch_dtype in DTYPE_MAPPING:
        return DTYPE_MAPPING[torch_dtype]

    # 尝试直接转换
    try:
        dtype_name = str(torch_dtype).split('.')[-1]
        return getattr(ms, dtype_name)
    except (AttributeError, ValueError):
        logger.warning(f"Cannot map PyTorch dtype {torch_dtype} to MindSpore dtype")
        return torch_dtype


def ms_dtype_to_torch_dtype(ms_dtype: Any) -> Any:
    """
    将MindSpore数据类型转换为PyTorch数据类型

    Args:
        ms_dtype: MindSpore数据类型

    Returns:
        PyTorch数据类型
    """
    if not MS_AVAILABLE or not TORCH_AVAILABLE:
        return ms_dtype

    # 创建反向映射
    reverse_mapping = {v: k for k, v in DTYPE_MAPPING.items()}
    if ms_dtype in reverse_mapping:
        return reverse_mapping[ms_dtype]

    # 尝试直接转换
    try:
        dtype_name = str(ms_dtype).split('.')[-1]
        # 处理MindSpore特有的命名
        if dtype_name == 'bool_':
            dtype_name = 'bool'
        return getattr(torch, dtype_name)
    except (AttributeError, ValueError):
        logger.warning(f"Cannot map MindSpore dtype {ms_dtype} to PyTorch dtype")
        return ms_dtype


def torch_device_to_ms_device(device: Union[str, Any]) -> str:
    """
    将PyTorch设备名称转换为MindSpore设备名称

    Args:
        device: PyTorch设备名称或设备对象

    Returns:
        MindSpore设备名称
    """
    device_str = str(device).lower()

    # 处理CUDA设备
    if device_str.startswith('cuda'):
        return 'GPU'

    # 处理CPU设备
    elif device_str.startswith('cpu'):
        return 'CPU'

    # 处理其他设备类型
    for torch_dev, ms_dev in DEVICE_MAPPING.items():
        if torch_dev in device_str:
            return ms_dev

    # 默认返回CPU
    return 'CPU'


def ensure_ms_tensor(tensor: Any) -> Any:
    """
    确保输入是MindSpore张量

    Args:
        tensor: 输入张量，可以是PyTorch张量、NumPy数组或MindSpore张量

    Returns:
        MindSpore张量
    """
    if not MS_AVAILABLE:
        return tensor

    if isinstance(tensor, ms.Tensor):
        return tensor

    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        # PyTorch -> NumPy -> MindSpore 转换
        numpy_tensor = tensor.detach().cpu().numpy()
        return ms.Tensor(numpy_tensor, dtype=torch_dtype_to_ms_dtype(tensor.dtype))

    # 尝试直接转换为MindSpore张量
    try:
        return ms.Tensor(tensor)
    except Exception as e:
        logger.warning(f"Failed to convert to MindSpore tensor: {e}")
        return tensor


def ensure_torch_tensor(tensor: Any) -> Any:
    """
    确保输入是PyTorch张量

    Args:
        tensor: 输入张量，可以是MindSpore张量、NumPy数组或PyTorch张量

    Returns:
        PyTorch张量
    """
    if not TORCH_AVAILABLE:
        return tensor

    if isinstance(tensor, torch.Tensor):
        return tensor

    if MS_AVAILABLE and isinstance(tensor, ms.Tensor):
        # MindSpore -> NumPy -> PyTorch 转换
        numpy_tensor = tensor.asnumpy()
        return torch.from_numpy(numpy_tensor)

    # 尝试直接转换为PyTorch张量
    try:
        return torch.tensor(tensor)
    except Exception as e:
        logger.warning(f"Failed to convert to PyTorch tensor: {e}")
        return tensor


def ms_to_torch_device_format(device: str) -> str:
    """
    将MindSpore设备格式转换为PyTorch设备格式

    Args:
        device: MindSpore设备名称

    Returns:
        PyTorch设备名称
    """
    device_lower = device.lower()
    if device_lower == 'gpu':
        return 'cuda'
    elif device_lower == 'cpu':
        return 'cpu'
    elif device_lower == 'npu':
        return 'npu'
    return 'cpu'


def safe_operation(func, *args, **kwargs):
    """
    安全执行操作，捕获并记录异常

    Args:
        func: 要执行的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数执行结果，如果失败则返回None
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {e}")
        return None


# 兼容性函数：提供与PyTorch API相似的接口但内部使用MindSpore
def ms_tensor(*args, **kwargs):
    """
    兼容性函数：创建MindSpore张量，接口与torch.tensor相似
    """
    if not MS_AVAILABLE:
        logger.error("MindSpore not available")
        return None

    # 处理dtype参数转换
    if 'dtype' in kwargs and TORCH_AVAILABLE:
        kwargs['dtype'] = torch_dtype_to_ms_dtype(kwargs['dtype'])

    return ms.Tensor(*args, **kwargs)


def ms_device(device: str = 'CPU'):
    """
    兼容性函数：获取设备，接口与torch.device相似
    """
    # MindSpore没有device对象，返回设备名称字符串
    return torch_device_to_ms_device(device)
