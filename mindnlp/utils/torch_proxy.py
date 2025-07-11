import sys
import types
import importlib
import importlib.metadata
from collections import defaultdict

class TorchProxyModule(types.ModuleType):
    """递归代理模块，支持任意深度的模块路径"""
    
    # 缓存已创建的代理模块
    _proxy_cache = defaultdict(dict)
    
    def __new__(cls, real_module, proxy_name):
        """使用缓存避免重复创建代理"""
        # 生成缓存键：真实模块ID + 代理名称
        cache_key = (id(real_module), proxy_name)
        
        # 如果已存在缓存，直接返回
        if cache_key in cls._proxy_cache[real_module]:
            return cls._proxy_cache[real_module][cache_key]
        
        # 创建新实例并缓存
        instance = super().__new__(cls, proxy_name)
        cls._proxy_cache[real_module][cache_key] = instance
        return instance
    
    def __init__(self, real_module, proxy_name):
        """初始化代理模块"""
        super().__init__(proxy_name)
        self._real_module = real_module
        self._proxy_name = proxy_name
        self._submodule_proxies = {}
        
        # 设置关键元数据
        self.__name__ = proxy_name
        self.__package__ = proxy_name
        self.__file__ = "<mindnlp-torch-proxy>"
        
    def __getattr__(self, name):
        """动态获取属性并创建子模块代理"""
        # 1. 尝试从真实模块获取属性
        try:
            real_attr = getattr(self._real_module, name)
        except AttributeError:
            raise AttributeError(
                f"module '{self._proxy_name}' has no attribute '{name}'"
            )

        # 2. 如果是模块类型，创建递归代理
        if isinstance(real_attr, types.ModuleType):
            # 构建子模块的代理名称
            sub_proxy_name = f"{self._proxy_name}.{name}"

            if name in self._submodule_proxies:
                return self._submodule_proxies[name]

            # 创建或获取子模块代理
            proxy_sub = TorchProxyModule(
                real_attr, 
                sub_proxy_name
            )

            self._submodule_proxies[name] = proxy_sub
            # 缓存子模块代理
            self._submodule_proxies[name] = proxy_sub
            # 注册到sys.modules
            sys.modules[sub_proxy_name] = proxy_sub
            # 注册到父模块
            setattr(self, name, proxy_sub)
            return self._submodule_proxies[name]
        
        # 4. 其他类型直接返回
        return real_attr
    
    def __setattr__(self, name, value):
        """处理属性设置"""
        # 内部属性直接设置
        if name in {"_real_module", "_proxy_name", "_submodule_proxies"}:
            super().__setattr__(name, value)
            return

        # 其他属性设置到真实模块
        if name not in self._submodule_proxies:
            setattr(self._real_module, name, value)

    def __dir__(self):
        """返回真实模块的属性列表"""
        return dir(self._real_module)
    
    def __repr__(self):
        """友好的代理模块表示"""
        return f"<proxy module '{self._proxy_name}' from '{self._real_module.__name__}'>"

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
    import mindnlp
    torch_proxy = TorchProxyModule(mindnlp.core, 'torch')
    sys.modules["torch"] = torch_proxy

    # 设置必要的元数据
    torch_proxy.__version__ = "2.1.1+dev"

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
                version="2.1.1+dev",
                metadata={"Name": "torch", "Version": "2.1.1+dev"},
                read_text=lambda f: f"Name: torch\nVersion: 2.1.1+dev" if f == "METADATA" else None
            )
        return orig_distribution(dist_name)
    
    # 确保分发列表中有 torch
    def patched_distributions(**kwargs):
        dists = list(orig_distributions(**kwargs))
        dists.append(types.SimpleNamespace(
            name="torch",
            version="2.1.1+dev",
            metadata={"Name": "torch", "Version": "2.1.1+dev"},
            files=[],
            locate_file=lambda p: None,
            _normalized_name='torch',
            entry_points=[]
        ))
        return dists
    
    # 应用补丁
    importlib.metadata.distribution = patched_distribution
    importlib.metadata.distributions = patched_distributions
