"""
Versioned Patch Registry

Provides a registry system for managing version-specific patches.
"""

from packaging import version, specifiers
from typing import Dict, List, Callable, Optional, NamedTuple
import warnings
import sys


class PatchInfo(NamedTuple):
    """补丁信息"""
    version_spec: str
    func: Callable
    priority: int
    depends_on: List[str]
    name: str
    description: str = ""


class VersionedPatchRegistry:
    """
    版本化补丁注册表
    
    支持：
    - 版本范围匹配（使用 packaging.specifiers）
    - 补丁优先级
    - 补丁依赖
    - 错误隔离
    """
    
    def __init__(self, library_name: str):
        self.library_name = library_name
        self._patches: List[PatchInfo] = []
        self._applied_patches: set = set()
    
    def register(self, 
                 version_spec: str,
                 patch_func: Callable,
                 priority: int = 0,
                 depends_on: Optional[List[str]] = None,
                 description: str = ""):
        """
        注册补丁
        
        Args:
            version_spec: 版本规范，如 ">=4.56.0,<4.57.0" 或 "~=4.56.0"
            patch_func: 补丁函数
            priority: 优先级（数字越大越先执行，默认0）
            depends_on: 依赖的其他补丁名称列表
            description: 补丁描述
        """
        patch_name = patch_func.__name__
        self._patches.append(PatchInfo(
            version_spec=version_spec,
            func=patch_func,
            priority=priority,
            depends_on=depends_on or [],
            name=patch_name,
            description=description or patch_name
        ))
    
    def apply_patches(self, current_version: str, verbose: bool = False):
        """
        根据当前版本应用匹配的补丁
        
        Args:
            current_version: 当前库版本
            verbose: 是否输出详细信息
        """
        try:
            current = version.Version(current_version)
        except version.InvalidVersion:
            warnings.warn(
                f"Invalid version '{current_version}' for {self.library_name}, "
                "skipping patches"
            )
            return
        
        # 筛选匹配版本的补丁
        applicable_patches = []
        for patch in self._patches:
            try:
                spec = specifiers.SpecifierSet(patch.version_spec)
                if current in spec:
                    applicable_patches.append(patch)
            except specifiers.InvalidSpecifier:
                warnings.warn(
                    f"Invalid version spec '{patch.version_spec}' for patch "
                    f"'{patch.name}', skipping"
                )
        
        if not applicable_patches:
            return
        
        # 按优先级排序（高优先级先执行）
        applicable_patches.sort(key=lambda x: x.priority, reverse=True)
        
        # 应用补丁（处理依赖）
        applied = set()
        failed = set()
        
        for patch in applicable_patches:
            # 检查依赖是否都已应用
            missing_deps = [dep for dep in patch.depends_on if dep not in applied]
            if missing_deps:
                warnings.warn(
                    f"Patch '{patch.name}' skipped: missing dependencies: {missing_deps}"
                )
                continue
            
            # 应用补丁
            try:
                patch.func()
                applied.add(patch.name)
                self._applied_patches.add(patch.name)
            except Exception as e:
                failed.add(patch.name)
                warnings.warn(
                    f"Failed to apply patch '{patch.name}': {e}",
                    stacklevel=2
                )
    
    def get_applied_patches(self):
        """获取已应用的补丁列表"""
        return list(self._applied_patches)


# 全局注册表
_transformers_registry = VersionedPatchRegistry("transformers")
_diffusers_registry = VersionedPatchRegistry("diffusers")
_safetensors_registry = VersionedPatchRegistry("safetensors")


def register_transformers_patch(version_spec: str, priority: int = 0, 
                                depends_on: Optional[List[str]] = None,
                                description: str = ""):
    """
    装饰器：注册 transformers 补丁
    
    使用示例:
        @register_transformers_patch(">=4.56.0,<4.57.0", priority=10)
        def patch_pre_trained_model():
            # 补丁逻辑
            pass
    """
    def decorator(func: Callable):
        _transformers_registry.register(
            version_spec=version_spec,
            patch_func=func,
            priority=priority,
            depends_on=depends_on,
            description=description or func.__doc__ or func.__name__
        )
        return func
    return decorator


def register_diffusers_patch(version_spec: str, priority: int = 0,
                            depends_on: Optional[List[str]] = None,
                            description: str = ""):
    """装饰器：注册 diffusers 补丁"""
    def decorator(func: Callable):
        _diffusers_registry.register(
            version_spec=version_spec,
            patch_func=func,
            priority=priority,
            depends_on=depends_on,
            description=description or func.__doc__ or func.__name__
        )
        return func
    return decorator


def register_safetensors_patch(version_spec: str, priority: int = 0,
                              depends_on: Optional[List[str]] = None,
                              description: str = ""):
    """装饰器：注册 safetensors 补丁"""
    def decorator(func: Callable):
        _safetensors_registry.register(
            version_spec=version_spec,
            patch_func=func,
            priority=priority,
            depends_on=depends_on,
            description=description or func.__doc__ or func.__name__
        )
        return func
    return decorator


def apply_transformers_patches(verbose: bool = False):
    """应用所有 transformers 补丁"""
    try:
        import transformers
        _transformers_registry.apply_patches(
            transformers.__version__,
            verbose=verbose
        )
    except ImportError:
        pass


def apply_diffusers_patches(verbose: bool = False):
    """应用所有 diffusers 补丁"""
    try:
        import diffusers
        _diffusers_registry.apply_patches(
            diffusers.__version__,
            verbose=verbose
        )
    except ImportError:
        pass


def apply_safetensors_patches(verbose: bool = False):
    """应用所有 safetensors 补丁"""
    try:
        import safetensors
        # safetensors 可能没有 __version__ 属性，尝试获取版本
        try:
            version_str = safetensors.__version__
        except AttributeError:
            # 如果没有版本号，使用一个默认版本让所有补丁都应用
            # 由于补丁使用 ">=0.0.0"，这会匹配所有版本
            version_str = "0.0.0"
        _safetensors_registry.apply_patches(
            version_str,
            verbose=verbose
        )
    except ImportError:
        pass


def apply_all_patches(verbose: bool = False):
    """应用所有补丁"""
    apply_safetensors_patches(verbose=verbose)
    apply_transformers_patches(verbose=verbose)
    apply_diffusers_patches(verbose=verbose)

