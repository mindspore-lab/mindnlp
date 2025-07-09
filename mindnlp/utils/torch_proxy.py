import sys
import types
import importlib
import importlib.metadata

class TorchProxyModule(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        # 保存真实模块的引用
        self._real_module = None
    
    def _load_real_module(self):
        """按需加载真实模块"""
        if self._real_module is None:
            # 尝试直接导入mindnlp.core作为torch
            self._real_module = importlib.import_module("mindnlp.core")
            # 添加必要的元数据属性
            self._real_module.__name__ = "torch"
            self._real_module.__package__ = "torch"
            self._real_module.__file__ = "<mindnlp-torch-proxy>"
        
        return self._real_module
    
    def __getattr__(self, name):
        """任何属性访问都重定向到真实模块"""
        # 处理特殊元数据属性
        if name in {"__name__", "__package__", "__file__"}:
            return getattr(self._load_real_module(), name)
            
        return getattr(self._load_real_module(), name)
    
    def __setattr__(self, name, value):
        """属性设置也重定向到真实模块"""
        # 跳过自身内部属性
        if name in {"_real_module", "__name__", "__package__", "__file__"}:
            super().__setattr__(name, value)
        else:
            setattr(self._load_real_module(), name, value)
    
    def __dir__(self):
        """返回真实模块的属性列表"""
        return dir(self._load_real_module())

    def __getattribute__(self, name):
        """特殊处理元数据相关属性"""
        if name == '__file__':
            return '<virtual torch module>'
        if name == '__package__':
            return 'torch'
        if name == '__spec__':
            return self._create_mock_spec()
        return super().__getattribute__(name)

def initialize_torch_proxy():

    torch_proxy = TorchProxyModule()
    sys.modules["torch"] = torch_proxy

    # 设置必要的元数据
    torch_proxy.__version__ = "2.1.1"

    return torch_proxy

def setup_metadata_patch():
    """解决 importlib.metadata 找不到 torch 的问题"""
    # 保存原始函数
    orig_distribution = importlib.metadata.distribution
    orig_distributions = importlib.metadata.distributions
    
    # 拦截对 torch 分发的查询
    def patched_distribution(dist_name):
        if dist_name == "torch":
            return types.SimpleNamespace(
                version="2.1.1",
                metadata={"Name": "torch", "Version": "2.1.1"},
                read_text=lambda f: f"Name: torch\nVersion: 2.1.1" if f == "METADATA" else None
            )
        return orig_distribution(dist_name)
    
    # 确保分发列表中有 torch
    def patched_distributions(**kwargs):
        dists = list(orig_distributions(**kwargs))
        dists.append(types.SimpleNamespace(
            name="torch",
            version="2.1.1",
            metadata={"Name": "torch", "Version": "2.1.1"},
            files=[],
            locate_file=lambda p: None,
            _normalized_name='torch',
            entry_points=[]
        ))
        return dists
    
    # 应用补丁
    importlib.metadata.distribution = patched_distribution
    importlib.metadata.distributions = patched_distributions
