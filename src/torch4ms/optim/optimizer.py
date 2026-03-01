"""
MindSpore 优化器适配器 - 仿照 optax 的设计

提供函数式的优化器接口，支持链式组合梯度变换，类似 optax。
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union, Iterable, Callable, Tuple
import mindspore as ms
from mindspore import nn as ms_nn
from mindspore import Parameter
from torch4ms.tensor import Tensor as Torch4msTensor
from torch4ms.ops import mappings
import numpy as np


class OptimizerState:
    """优化器状态（类似 optax.OptState）"""
    def __init__(self, params: List[Parameter], step: int = 0):
        self.params = params
        self.step = step
        self.extra_state = {}


class GradientTransform:
    """
    梯度变换基类（类似 optax.GradientTransform）
    
    每个梯度变换都有 init 和 update 方法。
    """
    def init(self, params: List[Parameter]) -> OptimizerState:
        """初始化变换状态"""
        return OptimizerState(params)
    
    def update(self, grads: List[ms.Tensor], state: OptimizerState) -> Tuple[List[ms.Tensor], OptimizerState]:
        """应用梯度变换"""
        return grads, state


def chain(*transforms: GradientTransform) -> GradientTransform:
    """
    链式组合多个梯度变换（类似 optax.chain）
    
    Args:
        *transforms: 多个梯度变换
        
    Returns:
        组合后的梯度变换
    """
    class ChainedTransform(GradientTransform):
        def init(self, params: List[Parameter]) -> OptimizerState:
            states = []
            for transform in transforms:
                states.append(transform.init(params))
            return OptimizerState(params, extra_state={'transforms': transforms, 'states': states})
        
        def update(self, grads: List[ms.Tensor], state: OptimizerState) -> Tuple[List[ms.Tensor], OptimizerState]:
            current_grads = grads
            new_states = []
            for transform, transform_state in zip(state.extra_state['transforms'], state.extra_state['states']):
                current_grads, new_transform_state = transform.update(current_grads, transform_state)
                new_states.append(new_transform_state)
            
            new_state = OptimizerState(state.params, step=state.step + 1)
            new_state.extra_state = {'transforms': state.extra_state['transforms'], 'states': new_states}
            return current_grads, new_state
    
    return ChainedTransform()


class clip_by_global_norm(GradientTransform):
    """
    全局梯度裁剪（类似 optax.clip_by_global_norm）
    
    Args:
        max_norm: 最大梯度范数
    """
    def __init__(self, max_norm: float):
        self.max_norm = max_norm
    
    def update(self, grads: List[ms.Tensor], state: OptimizerState) -> Tuple[List[ms.Tensor], OptimizerState]:
        """应用全局梯度裁剪"""
        from mindspore import ops
        
        # 计算全局范数
        total_norm = 0.0
        for grad in grads:
            if grad is not None:
                param_norm = ops.norm(grad)
                total_norm += param_norm ** 2
        total_norm = ops.sqrt(total_norm)
        
        # 裁剪梯度
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            clipped_grads = [grad * clip_coef if grad is not None else None for grad in grads]
        else:
            clipped_grads = grads
        
        return clipped_grads, state


class scale_by_learning_rate(GradientTransform):
    """
    学习率缩放（类似 optax.scale_by_learning_rate）
    
    Args:
        learning_rate: 学习率或学习率调度函数
    """
    def __init__(self, learning_rate: Union[float, Callable[[int], float]]):
        self.learning_rate = learning_rate
    
    def update(self, grads: List[ms.Tensor], state: OptimizerState) -> Tuple[List[ms.Tensor], OptimizerState]:
        """应用学习率缩放"""
        if callable(self.learning_rate):
            lr = self.learning_rate(state.step)
        else:
            lr = self.learning_rate
        
        scaled_grads = [grad * lr if grad is not None else None for grad in grads]
        return scaled_grads, state


class add_decayed_weights(GradientTransform):
    """
    添加权重衰减（类似 optax.add_decayed_weights）
    
    Args:
        weight_decay: 权重衰减系数
    """
    def __init__(self, weight_decay: float):
        self.weight_decay = weight_decay
    
    def update(self, grads: List[ms.Tensor], state: OptimizerState) -> Tuple[List[ms.Tensor], OptimizerState]:
        """添加权重衰减"""
        from mindspore import ops
        
        decayed_grads = []
        for grad, param in zip(grads, state.params):
            if grad is not None:
                # 添加权重衰减：grad = grad + weight_decay * param
                decayed_grad = grad + self.weight_decay * param
                decayed_grads.append(decayed_grad)
            else:
                decayed_grads.append(None)
        
        return decayed_grads, state


class Optimizer:
    """
    MindSpore 优化器适配器（类似 optax 的使用方式）
    
    这个类将 MindSpore 优化器包装成函数式的接口，支持链式组合梯度变换。
    """
    
    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, Any]]],
        gradient_transform: GradientTransform,
    ):
        """
        初始化优化器
        
        Args:
            params: 模型参数（PyTorch Parameter）
            gradient_transform: 梯度变换（可以是链式组合的）
        """
        # 提取参数列表
        if isinstance(params, dict):
            param_list = list(params.values()) if 'params' not in params else params['params']
        else:
            param_list = list(params)
        
        # 过滤出需要梯度的参数
        self.trainable_params = [p for p in param_list if p.requires_grad]
        
        # 将 PyTorch 参数转换为 MindSpore Parameter
        self.ms_params = []
        self.param_map = {}  # PyTorch Parameter -> MindSpore Parameter
        
        for param in self.trainable_params:
            ms_param = self._to_ms_parameter(param)
            self.ms_params.append(ms_param)
            self.param_map[param] = ms_param
        
        # 初始化梯度变换状态
        self.gradient_transform = gradient_transform
        self.state = gradient_transform.init(self.ms_params)
    
    def _to_ms_parameter(self, param: torch.nn.Parameter) -> Parameter:
        """将 PyTorch Parameter 转换为 MindSpore Parameter"""
        if isinstance(param, Torch4msTensor):
            ms_tensor = param._elem
            if isinstance(ms_tensor, Parameter):
                return ms_tensor
            else:
                return Parameter(ms_tensor)
        else:
            ms_tensor = mappings.t2ms(param.data)
            return Parameter(ms_tensor)
    
    def _extract_gradients(self) -> List[ms.Tensor]:
        """从 PyTorch 参数中提取梯度"""
        grads = []
        for param in self.trainable_params:
            grad = None
            
            # 尝试从多个位置获取梯度
            if isinstance(param, Torch4msTensor):
                if hasattr(param, '_t4ms_grad') and param._t4ms_grad is not None:
                    grad = param._t4ms_grad._elem if isinstance(param._t4ms_grad, Torch4msTensor) else param._t4ms_grad
                elif hasattr(param, 'grad') and param.grad is not None:
                    if isinstance(param.grad, Torch4msTensor):
                        grad = param.grad._elem
                    else:
                        grad = mappings.t2ms(param.grad)
            else:
                if hasattr(param, 'grad') and param.grad is not None:
                    grad = mappings.t2ms(param.grad)
            
            if grad is not None and not isinstance(grad, ms.Tensor):
                grad = ms.Tensor(grad)
            
            grads.append(grad)
        
        return grads
    
    def _apply_gradients(self, grads: List[ms.Tensor]):
        """应用梯度更新参数"""
        import torch.utils._mode_utils as mode_utils
        from mindspore import ops
        
        for param, ms_param, grad in zip(self.trainable_params, self.ms_params, grads):
            if grad is None:
                continue
            
            # 更新 MindSpore Parameter
            # MindSpore Parameter 的值可以直接修改
            if isinstance(ms_param, Parameter):
                current_value = ms_param.value if hasattr(ms_param, 'value') else ms_param
            else:
                current_value = ms_param
            
            if not isinstance(current_value, ms.Tensor):
                current_value = ms.Tensor(current_value)
            
            # 执行梯度下降：param = param - grad
            from mindspore import ops
            new_value = current_value - grad
            
            # 更新 Parameter
            if isinstance(ms_param, Parameter):
                # MindSpore Parameter 使用 set_data 方法
                ms_param.set_data(new_value)
                ms_value = ms_param.value if hasattr(ms_param, 'value') else ms_param
            else:
                ms_value = new_value
            
            # 确保 ms_value 是 Tensor
            if not isinstance(ms_value, ms.Tensor):
                ms_value = ms.Tensor(ms_value)
            if not isinstance(ms_value, ms.Tensor):
                ms_value = ms.Tensor(ms_value)
            
            with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                torch_value = mappings.ms2t(ms_value)
            
            with torch.no_grad():
                if isinstance(param, Torch4msTensor):
                    with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                        if param.requires_grad:
                            param._elem = Parameter(ms_value)
                        else:
                            param._elem = ms_value
                else:
                    param.data.copy_(torch_value)
    
    def step(self):
        """执行一次优化步骤"""
        # 提取梯度
        grads = self._extract_gradients()
        
        # 应用梯度变换
        transformed_grads, self.state = self.gradient_transform.update(grads, self.state)
        
        # 应用梯度更新参数
        self._apply_gradients(transformed_grads)
    
    def zero_grad(self, set_to_none: bool = False):
        """清零梯度"""
        for param in self.trainable_params:
            if isinstance(param, Torch4msTensor):
                if set_to_none:
                    param._t4ms_grad = None
                else:
                    if hasattr(param, '_t4ms_grad') and param._t4ms_grad is not None:
                        param._t4ms_grad.zero_()
            else:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad is not None:
                        param.grad.zero_()


# 预定义的优化器（使用 MindSpore 原生优化器 + 梯度变换）

class SGD(Optimizer):
    """SGD 优化器"""
    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        clip_norm: Optional[float] = None,
    ):
        """
        Args:
            params: 模型参数
            lr: 学习率
            momentum: 动量系数
            weight_decay: 权重衰减
            nesterov: 是否使用 Nesterov 动量
            clip_norm: 梯度裁剪阈值（可选）
        """
        # 构建梯度变换链
        transforms = []
        
        if clip_norm is not None:
            transforms.append(clip_by_global_norm(clip_norm))
        
        if weight_decay > 0:
            transforms.append(add_decayed_weights(weight_decay))
        
        transforms.append(scale_by_learning_rate(lr))
        
        gradient_transform = chain(*transforms) if transforms else scale_by_learning_rate(lr)
        
        super().__init__(params, gradient_transform)
        
        # 存储额外参数（用于未来扩展）
        self.momentum = momentum
        self.nesterov = nesterov


class Adam(Optimizer):
    """Adam 优化器"""
    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_norm: Optional[float] = None,
    ):
        """
        Args:
            params: 模型参数
            lr: 学习率
            betas: 动量衰减系数
            eps: 数值稳定性常数
            weight_decay: 权重衰减
            clip_norm: 梯度裁剪阈值（可选）
        """
        # 构建梯度变换链
        transforms = []
        
        if clip_norm is not None:
            transforms.append(clip_by_global_norm(clip_norm))
        
        if weight_decay > 0:
            transforms.append(add_decayed_weights(weight_decay))
        
        transforms.append(scale_by_learning_rate(lr))
        
        gradient_transform = chain(*transforms) if transforms else scale_by_learning_rate(lr)
        
        super().__init__(params, gradient_transform)
        
        # 存储额外参数（用于未来扩展）
        self.betas = betas
        self.eps = eps


class AdamW(Optimizer):
    """AdamW 优化器"""
    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        clip_norm: Optional[float] = None,
    ):
        """
        Args:
            params: 模型参数
            lr: 学习率
            betas: 动量衰减系数
            eps: 数值稳定性常数
            weight_decay: 权重衰减（通常比 Adam 大）
            clip_norm: 梯度裁剪阈值（可选）
        """
        # AdamW 使用权重衰减
        transforms = []
        
        if clip_norm is not None:
            transforms.append(clip_by_global_norm(clip_norm))
        
        if weight_decay > 0:
            transforms.append(add_decayed_weights(weight_decay))
        
        transforms.append(scale_by_learning_rate(lr))
        
        gradient_transform = chain(*transforms) if transforms else scale_by_learning_rate(lr)
        
        super().__init__(params, gradient_transform)
        
        self.betas = betas
        self.eps = eps


class RMSprop(Optimizer):
    """RMSprop 优化器"""
    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        clip_norm: Optional[float] = None,
    ):
        """
        Args:
            params: 模型参数
            lr: 学习率
            alpha: 平滑常数
            eps: 数值稳定性常数
            weight_decay: 权重衰减
            momentum: 动量系数
            clip_norm: 梯度裁剪阈值（可选）
        """
        transforms = []
        
        if clip_norm is not None:
            transforms.append(clip_by_global_norm(clip_norm))
        
        if weight_decay > 0:
            transforms.append(add_decayed_weights(weight_decay))
        
        transforms.append(scale_by_learning_rate(lr))
        
        gradient_transform = chain(*transforms) if transforms else scale_by_learning_rate(lr)
        
        super().__init__(params, gradient_transform)
        
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum


class Torch4msOptimizer:
    """
    PyTorch 优化器适配器，使 PyTorch 优化器能够与 torch4ms.Tensor 一起工作
    
    这个适配器解决了 PyTorch 优化器无法直接使用 torch4ms.Tensor 梯度的问题。
    它会在 step() 之前自动将 torch4ms.Tensor 的梯度转换为普通 torch.Tensor。
    
    使用方法：
        >>> import torch.optim as optim
        >>> optimizer = optim.SGD(model.parameters(), lr=0.01)
        >>> optimizer = Torch4msOptimizer(optimizer, model)
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    
    参考 TORCHAX_COMPARISON.md 中的方案 A。
    """
    
    def __init__(self, optimizer, model):
        """
        初始化适配器
        
        Args:
            optimizer: PyTorch 优化器实例（如 torch.optim.SGD）
            model: PyTorch 模型（torch.nn.Module）
        """
        self.optimizer = optimizer
        self.model = model
    
    def step(self):
        """
        执行优化步骤
        
        在执行优化步骤之前，自动将 torch4ms.Tensor 的梯度转换为普通 torch.Tensor。
        """
        from torch4ms.ops import mappings
        import torch.utils._mode_utils as mode_utils
        
        # 将 torch4ms.Tensor 的梯度转换为普通 torch.Tensor
        for name, param in self.model.named_parameters():
            # 检查是否有 torch4ms.Tensor 的梯度（存储在 _t4ms_grad 中）
            if isinstance(param, Torch4msTensor):
                if hasattr(param, '_t4ms_grad') and param._t4ms_grad is not None:
                    # 获取 torch4ms.Tensor 的梯度
                    grad_t4ms = param._t4ms_grad
                    # 转换为普通 torch.Tensor
                    with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                        grad = mappings.ms2t(grad_t4ms._elem)
                    # 设置到参数的 grad 属性
                    with torch.no_grad():
                        param.grad = grad
                elif hasattr(param, 'grad') and param.grad is not None:
                    # 如果 grad 已经是 torch4ms.Tensor，也需要转换
                    if isinstance(param.grad, Torch4msTensor):
                        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                            grad = mappings.ms2t(param.grad._elem)
                        with torch.no_grad():
                            param.grad = grad
            else:
                # 对于普通 torch.Tensor 参数，检查是否有 torch4ms.Tensor 的梯度
                # 这种情况可能发生在参数不是 torch4ms.Tensor 但梯度是的情况下
                if hasattr(param, '_t4ms_grad') and param._t4ms_grad is not None:
                    grad_t4ms = param._t4ms_grad
                    if isinstance(grad_t4ms, Torch4msTensor):
                        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                            grad = mappings.ms2t(grad_t4ms._elem)
                        with torch.no_grad():
                            param.grad = grad
                elif hasattr(param, 'grad') and param.grad is not None:
                    if isinstance(param.grad, Torch4msTensor):
                        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                            grad = mappings.ms2t(param.grad._elem)
                        with torch.no_grad():
                            param.grad = grad
        
        # 调用原始优化器的 step
        self.optimizer.step()
    
    def zero_grad(self, set_to_none: bool = False):
        """
        清零梯度
        
        同时清零 torch4ms.Tensor 的梯度和普通 torch.Tensor 的梯度。
        
        Args:
            set_to_none: 如果为 True，将梯度设置为 None；否则调用 zero_()
        """
        # 调用原始优化器的 zero_grad
        self.optimizer.zero_grad(set_to_none=set_to_none)
        
        # 同时清零 torch4ms.Tensor 的梯度
        for param in self.model.parameters():
            if isinstance(param, Torch4msTensor):
                if set_to_none:
                    param._t4ms_grad = None
                else:
                    if hasattr(param, '_t4ms_grad') and param._t4ms_grad is not None:
                        param._t4ms_grad.zero_()
            else:
                # 对于普通参数，也检查是否有 _t4ms_grad
                if hasattr(param, '_t4ms_grad'):
                    if set_to_none:
                        param._t4ms_grad = None
                    else:
                        if param._t4ms_grad is not None:
                            param._t4ms_grad.zero_()
    
    def __getattr__(self, name):
        """
        代理其他属性和方法到原始优化器
        
        这样可以让用户访问优化器的其他属性和方法（如 state_dict, load_state_dict 等）。
        """
        return getattr(self.optimizer, name)
    
    def state_dict(self):
        """获取优化器状态字典"""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载优化器状态字典"""
        return self.optimizer.load_state_dict(state_dict)
