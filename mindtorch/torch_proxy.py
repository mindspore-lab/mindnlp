import sys
import types
import importlib
import importlib.metadata
import importlib.abc
import importlib.machinery
from types import ModuleType

from mindtorch.configs import DEVICE_TARGET

TORCH_VERSION = '2.7.1+dev'

# mindtorch/__init__.py
import sys
import importlib.abc
import importlib.util

class MindTorchFinder(importlib.abc.MetaPathFinder):
    """
    自定义查找器，用于拦截对 'torch' 模块的导入。
    """
    def find_spec(self, fullname, path, target=None):
        # 仅当导入的模块名为 'torch' 时进行拦截
        if fullname == "torch" or fullname == 'torch_npu' or fullname.startswith("torch."):
            # 创建一个模块规范，并指定由自定义的加载器来加载
            # 注意：这里的 `__name__` 是 'mindtorch'，我们需要返回代表 'torch' 的规范
            spec = importlib.util.spec_from_loader(fullname, MindTorchLoader())
            return spec
        # 对于其他模块，不进行处理，交由其他查找器
        return None


class MindTorchLoader(importlib.abc.Loader):
    """
    自定义加载器，当导入 'torch' 时，返回 mindtorch 模块对象。
    """
    def create_module(self, spec):
        """
        创建模块对象。这里直接返回已导入的 mindtorch 模块本身。
        """
        fullname = spec.name
        
        if fullname == "torch":
            # 顶层 torch 模块直接返回 mindtorch
            return sys.modules["mindtorch"]
        elif fullname == 'torch_npu':
            return importlib.import_module('mindtorch.npu')

        # 处理子模块：将 torch.xxx 映射到 mindtorch.xxx
        submodule_name = fullname.replace("torch.", "mindtorch.", 1)
        
        try:
            # 尝试导入对应的 mindtorch 子模块
            submodule = importlib.import_module(submodule_name)
            return submodule
        except ImportError:
            # 如果 mindtorch 没有对应的子模块，可以选择返回 None 或创建虚拟模块
            return None

    def exec_module(self, module):
        """
        执行模块。因为 'torch' 模块实际上是已经加载好的 mindtorch 模块，
        所以这里不需要再执行额外的初始化代码。
        重要：避免重新执行模块代码，否则可能导致状态重置或无限递归。
        """
        # 可以在这里添加一些日志或轻量级的检查，但通常留空
        pass

# 将自定义查找器添加到 sys.meta_path 的开头，使其具有最高优先级

# 以下是你的 mindtorch 库的原有代码和接口...
# 例如：from .tensor import Tensor, ...
# 确保 mindtorch 的 API 与 PyTorch 保持一致。
def initialize_torch_proxy():
    sys.meta_path.insert(0, MindTorchFinder())
    import torch
    torch.__version__ = TORCH_VERSION



def setup_metadata_patch():
    """解决 importlib.metadata 找不到 torch 的问题"""
    # 保存原始函数
    orig_distribution = importlib.metadata.distribution
    orig_distributions = importlib.metadata.distributions

    # 拦截对 torch 分发的查询
    def patched_distribution(dist_name):
        if dist_name == "torch":
            return types.SimpleNamespace(
                version=TORCH_VERSION,
                metadata={"Name": "torch", "Version": TORCH_VERSION},
                read_text=lambda f: (
                    f"Name: torch\nVersion: {TORCH_VERSION}" if f == "METADATA" else None
                ),
            )
        return orig_distribution(dist_name)

    # 确保分发列表中有 torch
    def patched_distributions(**kwargs):
        dists = list(orig_distributions(**kwargs))
        dists.append(
            types.SimpleNamespace(
                name="torch",
                version=TORCH_VERSION,
                metadata={"Name": "torch", "Version": TORCH_VERSION},
                files=[],
                locate_file=lambda p: None,
                _normalized_name="torch",
                entry_points=[],
            )
        )
        return dists

    # 应用补丁
    importlib.metadata.distribution = patched_distribution
    importlib.metadata.distributions = patched_distributions
