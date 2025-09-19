import sys
import types
import importlib
import importlib.metadata
import importlib.abc
import importlib.machinery
from types import ModuleType

from mindtorch.configs import DEVICE_TARGET

TORCH_VERSION = '2.7.1+dev'

class RedirectFinder(importlib.abc.MetaPathFinder):
    def __init__(self, redirect_map):
        # 重定向规则：被代理模块 -> 实际模块
        self.redirect_map = redirect_map

    def find_spec(self, fullname, path, target=None):
        # 识别需要重定向的模块
        for proxy_prefix, target_prefix in self.redirect_map.items():
            if fullname == proxy_prefix or fullname.startswith(proxy_prefix + "."):
                # 计算实际模块名
                target_name = fullname.replace(proxy_prefix, target_prefix, 1)
                if DEVICE_TARGET == 'Ascend':
                    target_name = target_name.replace('cuda', 'npu')
                try:
                    importlib.import_module(target_name)
                except Exception as e:
                    raise e

                return importlib.machinery.ModuleSpec(
                    name=fullname,
                    loader=RedirectLoader(target_name),
                    is_package=self._is_package(target_name),
                )
        return None

    def _is_package(self, module_name):
        # 检测模块是否为包（包含子模块）
        try:
            module = importlib.import_module(module_name)
            return hasattr(module, "__path__")
        except ImportError:
            return False


class RedirectLoader(importlib.abc.Loader):
    def __init__(self, target_name):
        self.target_name = target_name

    def create_module(self, spec):
        # 创建代理模块对象
        module = ModuleType(spec.name)
        module.__spec__ = spec
        module.__path__ = []
        module.__loader__ = self
        module.__package__ = spec.name
        return module

    def exec_module(self, module):
        # 动态设置__class__以代理属性访问
        class ProxyModule(type(module)):
            def __getattr__(_, name):
                # 动态导入实际模块中的属性
                if DEVICE_TARGET == 'Ascend':
                    name = name.replace('cuda', 'npu')
                try:
                    target_module = importlib.import_module(self.target_name)
                except ImportError as e:
                    raise AttributeError(f"Target module {self.target_name} could not be imported: {e}") from e
                except Exception as e:
                    raise e

                # 处理子模块导入 (e.g. torch.nn -> mindtorch.nn)
                if hasattr(target_module, name):
                    return getattr(target_module, name)

                # 处理从子模块导入 (e.g. from torch.nn import Module)
                try:
                    submodule_name = f"{self.target_name}.{name}"
                    return importlib.import_module(submodule_name)
                except ImportError as e:
                    raise AttributeError(
                        f"Module '{self.target_name}' has no attribute '{name}'"
                    )

            def __setattr__(_, name, value):
                try:
                    target_module = importlib.import_module(self.target_name)
                    if not hasattr(target_module, name):
                        return
                except Exception as e:
                    raise e
                return super().__setattr__(name, value)

        # 继承原始模块的特殊属性
        module.__class__ = ProxyModule


# 配置重定向规则
REDIRECT_MAP = {
    "torch": "mindtorch",
}
if DEVICE_TARGET == 'Ascend':
    REDIRECT_MAP["torch_npu"] = 'mindtorch.npu'

def initialize_torch_proxy():
    sys.meta_path.insert(0, RedirectFinder(REDIRECT_MAP))
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
