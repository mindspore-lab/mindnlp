"""
从 torch.nn.Module 提取用于 autograd 的 forward 函数

这个模块提供了 AutogradForwardExtractor 类，用于将状态化的 PyTorch 模块
转换为函数式的 forward 函数，以便与 MindSpore 的 GradOperation 一起使用。
"""
from typing import Callable, Dict, List, Tuple
import torch
from torch.nn.utils import stateless as torch_stateless
from torch4ms.tensor import Tensor as Torch4msTensor


def _is_trainable_param(value) -> bool:
    if isinstance(value, torch.nn.Parameter):
        return value.requires_grad
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, Torch4msTensor):
        return value.requires_grad
    return False


def _is_tensor_like(value) -> bool:
    return isinstance(value, (torch.Tensor, torch.nn.Parameter, Torch4msTensor))


def extract_all_buffers(module: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    深度遍历模块属性，提取参数与 buffers。

    与 torchax 的 JittableModule 类似，会扫描模块属性与子模块，
    以捕获非注册的参数/缓冲区以及共享权重别名。
    """
    params: Dict[str, torch.Tensor] = {}
    buffers: Dict[str, torch.Tensor] = {}

    def extract_one(m: torch.nn.Module, prefix: str) -> None:
        for key in dir(m):
            try:
                value = getattr(m, key)
            except Exception:
                continue
            qual_name = f"{prefix}{key}"
            if _is_trainable_param(value):
                params[qual_name] = value
            elif _is_tensor_like(value):
                buffers[qual_name] = value
        for name, child in m.named_children():
            extract_one(child, f"{prefix}{name}.")

    extract_one(module, "")
    return params, buffers


class AutogradForwardExtractor:
    """
    从 torch.nn.Module 提取用于 autograd 的 forward 函数
    
    功能：
    1. 提取可训练参数和 buffers
    2. 构建函数式 forward 函数
    3. 管理参数顺序（用于 GradOperation）
    
    示例：
        >>> model = torch.nn.Linear(10, 1)
        >>> extractor = AutogradForwardExtractor(model)
        >>> forward_fn = extractor.get_forward_fn()
        >>> # forward_fn 接受 (weight, bias, *input_args) 作为参数
    """
    
    def __init__(self, module: torch.nn.Module, dedup_parameters: bool = True):
        """
        初始化提取器
        
        Args:
            module: PyTorch 模块
        """
        self.module = module

        # 提取参数和 buffers（使用 named_parameters 确保顺序一致）
        # 先使用 named_parameters 获取顺序
        self.params = dict(module.named_parameters())
        self.buffers = dict(module.named_buffers())

        # 处理共享权重别名
        self._extra_dumped_weights: Dict[str, List[str]] = {}
        if dedup_parameters:
            seen: Dict[int, str] = {}
            for name, value in list(self.params.items()):
                value_id = id(value)
                if value_id not in seen:
                    seen[value_id] = name
                else:
                    base = seen[value_id]
                    self._extra_dumped_weights.setdefault(base, []).append(name)
                    del self.params[name]

        # 参数顺序（用于 GradOperation，保持与 named_parameters 一致）
        self.param_order = list(self.params.keys())
        self.buffer_order = list(self.buffers.keys())

        # 缓存 forward 函数
        self._forward_fn = None
        self._forward_fn_with_buffers = None
    
    def get_forward_fn(self, include_buffers: bool = False) -> Callable:
        """
        获取函数式 forward 函数
        
        Args:
            include_buffers: 是否将 buffers 作为参数传入 forward 函数
                           - True: forward_fn(*param_tensors, *buffer_tensors, *input_tensors)
                           - False: forward_fn(*param_tensors, *input_tensors)，buffers 使用模块中的值
        
        Returns:
            forward_fn: 函数式 forward 函数
        """
        if include_buffers:
            if self._forward_fn_with_buffers is None:
                self._forward_fn_with_buffers = self._build_forward_fn(include_buffers=True)
            return self._forward_fn_with_buffers
        else:
            if self._forward_fn is None:
                self._forward_fn = self._build_forward_fn(include_buffers=False)
            return self._forward_fn

    def get_ms_forward_fn(self, env, include_buffers: bool = False) -> Callable:
        """
        获取用于 MindSpore GradOperation 的 forward 函数。

        输入为 MindSpore Tensor，内部会转换为 torch.Tensor 执行前向（因为 functional_call 需要），
        并将结果转换回 MindSpore Tensor。
        """
        base_forward = self.get_forward_fn(include_buffers=include_buffers)

        def forward_fn(*ms_args):
            # 将 MindSpore Tensor 转为 torch.Tensor（functional_call 需要）
            # 先转为 torch4ms.Tensor，再转为 torch.Tensor
            t4_args = env.ms2t_iso(ms_args)
            torch_args = []
            for arg in t4_args:
                if isinstance(arg, Torch4msTensor):
                    # 转换为 torch.Tensor（用于 functional_call）
                    from torch4ms.ops import mappings
                    torch_args.append(mappings.ms2t(arg._elem))
                else:
                    torch_args.append(arg)
            
            # 调用 base_forward（它期望 torch.Tensor）
            with env:
                res = base_forward(*torch_args)
            
            # 将结果转换回 MindSpore Tensor
            if isinstance(res, torch.Tensor):
                from torch4ms.ops import mappings
                return mappings.t2ms(res)
            elif isinstance(res, Torch4msTensor):
                return res._elem
            else:
                return res

        return forward_fn
    
    def _build_forward_fn(self, include_buffers: bool) -> Callable:
        """
        构建函数式 forward 函数
        
        Args:
            include_buffers: 是否将 buffers 作为参数传入
        
        Returns:
            forward_fn: 函数式 forward 函数
        """
        param_order = self.param_order
        buffer_order = self.buffer_order
        module = self.module
        
        def forward_fn(*args):
            """
            函数式 forward 函数
            
            参数顺序：
            - 如果 include_buffers=True: (*param_tensors, *buffer_tensors, *input_tensors)
            - 如果 include_buffers=False: (*param_tensors, *input_tensors)
            """
            # 解析参数
            if include_buffers:
                n_params = len(param_order)
                n_buffers = len(buffer_order)
                
                # 提取参数和 buffers
                param_dict = dict(zip(param_order, args[:n_params]))
                buffer_dict = dict(zip(buffer_order, args[n_params:n_params+n_buffers]))
                input_args = args[n_params+n_buffers:]
            else:
                n_params = len(param_order)
                
                # 提取参数
                param_dict = dict(zip(param_order, args[:n_params]))
                buffer_dict = self.buffers  # 使用模块的 buffers（不计算梯度）
                input_args = args[n_params:]
            
            # 还原共享权重别名
            if self._extra_dumped_weights:
                for base_key, aliases in self._extra_dumped_weights.items():
                    for alias_key in aliases:
                        param_dict[alias_key] = param_dict[base_key]

            # 合并参数和 buffers
            params_and_buffers = {**param_dict, **buffer_dict}
            
            # 使用 functional_call 调用模块
            with torch_stateless._reparametrize_module(module, params_and_buffers):
                return module.forward(*input_args)
        
        return forward_fn
    
    def get_trainable_params(self) -> List[torch.Tensor]:
        """
        获取需要梯度的参数列表（torch4ms.Tensor）
        
        注意：这里返回的是原始 torch 参数（可能是 torch.nn.Parameter 或 torch.Tensor）。
        """
        return [self.params[name] for name in self.param_order]
    
    def get_param_names(self) -> List[str]:
        """
        获取参数名称列表（按顺序）
        
        Returns:
            参数名称列表
        """
        return self.param_order.copy()
    
    def get_buffer_names(self) -> List[str]:
        """
        获取 buffer 名称列表（按顺序）
        
        Returns:
            buffer 名称列表
        """
        return self.buffer_order.copy()

