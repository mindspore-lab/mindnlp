"""
MindSpore Autograd Function - 基于 torchax VJP 机制的设计

核心思路：
1. 在前向传播时，使用 MindSpore 的 GradOperation 生成梯度函数
2. 将梯度函数和必要信息保存到 context
3. 在反向传播时，恢复并调用梯度函数

这与 torchax 的 j2t_autograd 机制类似，但适配 MindSpore 的 API。
"""
import warnings
from typing import Any, Callable, Optional, Tuple, List
import torch
from torch.autograd import Function
import mindspore as ms
from mindspore import Tensor as ms_Tensor
from mindspore import ops
from mindspore.ops import GradOperation
import pickle
import io
from torch4ms.tensor import Tensor
from torch4ms.autograd.forward_extractor import AutogradForwardExtractor


class MindSporeFunction(Function):
    """
    MindSpore 版本的 torch.autograd.Function
    
    类似于 torchax 的 JaxFun，但使用 MindSpore 的 GradOperation。
    
    工作流程：
    1. forward: 执行前向计算，使用 GradOperation 生成梯度函数，保存到 context
    2. backward: 从 context 恢复梯度函数，调用它计算梯度
    """
    
    @staticmethod
    def forward(ctx, forward_fn, grad_fn_spec, *args):
        """
        前向传播：执行计算并准备反向传播函数
        
        Args:
            ctx: PyTorch autograd context
            forward_fn: MindSpore 前向函数（纯函数）
            grad_fn_spec: 梯度函数规格（用于序列化）
            *args: 前向传播的参数（需要梯度的张量）
        
        Returns:
            前向传播的输出
        """
        # 分离需要梯度的张量和其他参数
        # 这里假设所有 args 都是需要梯度的张量
        ms_args = tuple(arg._elem if isinstance(arg, Tensor) else arg for arg in args)
        
        # 获取环境（如果 args 中有 Tensor，从第一个获取环境）
        env = args[0]._env if args and isinstance(args[0], Tensor) else None
        if env is None:
            import torch4ms
            env = torch4ms.default_env()
        
        # 执行前向计算（确保在 torch4ms 环境中，以便 loss_fn 能被正确拦截）
        with env:
            output_ms = forward_fn(*ms_args)
        
        # 使用 GradOperation 生成梯度函数
        # 注意：MindSpore 的 GradOperation 需要在前向时准备好
        grad_op = GradOperation(get_all=True, sens_param=True)
        grad_fn = grad_op(forward_fn)
        
        # 序列化梯度函数（MindSpore 的 grad_fn 可能无法直接序列化）
        # 所以我们保存 forward_fn 和参数信息，在 backward 时重新生成
        ctx.forward_fn = forward_fn
        ctx.grad_fn_spec = grad_fn_spec
        ctx.num_inputs = len(args)
        ctx.save_for_backward(*args)  # 保存输入用于 backward
        
        # 将输出包装为 torch4ms.Tensor
        # 注意：torch.autograd.Function.apply 会自动设置 grad_fn
        # 但我们需要确保返回的 Tensor 能够正确触发 backward
        if isinstance(output_ms, ms_Tensor):
            # 需要获取环境，从第一个输入获取
            env = args[0]._env if args and isinstance(args[0], Tensor) else None
            if env is None:
                import torch4ms
                env = torch4ms.default_env()
            output = Tensor(output_ms, env, requires_grad=True)
            # 保存必要信息到 Tensor，以便 backward 时使用
            # 注意：这里我们依赖 torch.autograd.Function 的机制
            # apply 方法会自动将 MindSporeFunction 设置为 grad_fn
        else:
            output = output_ms
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用保存的梯度函数计算梯度
        
        Args:
            ctx: PyTorch autograd context
            grad_output: 输出梯度
        
        Returns:
            输入梯度元组
        """
        # 恢复输入
        saved_inputs = ctx.saved_tensors
        
        # 恢复梯度函数（重新生成，因为 MindSpore 的 grad_fn 可能无法序列化）
        forward_fn = ctx.forward_fn
        grad_op = GradOperation(get_all=True, sens_param=True)
        grad_fn = grad_op(forward_fn)
        
        # 转换输入为 MindSpore 张量
        ms_inputs = tuple(
            inp._elem if isinstance(inp, Tensor) else inp 
            for inp in saved_inputs
        )
        
        # 转换输出梯度为 MindSpore 张量
        grad_output_ms = grad_output._elem if isinstance(grad_output, Tensor) else grad_output
        
        # 调用梯度函数计算梯度
        try:
            input_grads_ms = grad_fn(*ms_inputs, grad_output_ms)
        except Exception as e:
            warnings.warn(f"GradOperation failed in backward: {e}")
            return tuple(None for _ in saved_inputs)
        
        # 转换梯度为 torch4ms.Tensor
        env = saved_inputs[0]._env if saved_inputs and isinstance(saved_inputs[0], Tensor) else None
        if env is None:
            import torch4ms
            env = torch4ms.default_env()
        
        input_grads = []
        for grad_ms in input_grads_ms:
            if grad_ms is not None:
                input_grads.append(Tensor(grad_ms, env, requires_grad=False))
            else:
                input_grads.append(None)
        
        return (None, None) + tuple(input_grads)  # None for forward_fn and grad_fn_spec


def ms2t_autograd(forward_fn: Callable, *args, **kwargs):
    """
    将 MindSpore 函数包装为 PyTorch autograd 函数
    
    类似于 torchax 的 j2t_autograd，但使用 MindSpore。
    
    Args:
        forward_fn: MindSpore 前向函数（纯函数，接受 MindSpore 张量）
        *args: 前向传播的参数（torch4ms.Tensor）
        **kwargs: 其他参数
    
    Returns:
        前向传播的输出（torch4ms.Tensor）
    """
    # 创建梯度函数规格（用于序列化，虽然我们实际上会重新生成）
    grad_fn_spec = {
        'forward_fn_id': id(forward_fn),
        'num_inputs': len(args)
    }
    
    # 调用 MindSporeFunction
    return MindSporeFunction.apply(forward_fn, grad_fn_spec, *args)


class ModuleOutputWrapper:
    """
    包装模块输出，保存 forward_fn 和 inputs 信息，用于自动 backward
    """
    def __init__(self, output: Tensor, forward_fn: Callable, inputs: Tuple[Tensor, ...]):
        self.output = output
        self.forward_fn = forward_fn
        self.inputs = inputs
    
    def sum(self, dim=None, keepdim=False):
        """对输出求和，并保存 forward_fn 信息"""
        from mindspore import ops
        
        # 创建一个包含 sum 的 forward_fn
        original_forward_fn = self.forward_fn
        
        def forward_fn_with_sum(*ms_params):
            output_ms = original_forward_fn(*ms_params)
            if dim is None:
                return ops.reduce_sum(output_ms)
            else:
                return ops.reduce_sum(output_ms, axis=dim, keep_dims=keepdim)
        
        if dim is None:
            result = self.output.sum()
        else:
            result = self.output.sum(dim=dim, keepdim=keepdim)
        # 保存包含 sum 的 forward_fn 和 inputs 到结果 Tensor
        result._module_forward_fn = forward_fn_with_sum
        result._module_inputs = self.inputs
        return result
    
    def __getattr__(self, name):
        """代理其他属性到 output"""
        return getattr(self.output, name)
    
    def __add__(self, other):
        result = self.output + other
        if hasattr(result, '_module_forward_fn'):
            result._module_forward_fn = self.forward_fn
            result._module_inputs = self.inputs
        else:
            result._module_forward_fn = self.forward_fn
            result._module_inputs = self.inputs
        return result
    
    def backward(self, **kwargs):
        """自动使用保存的 forward_fn 和 inputs"""
        if 'inputs' not in kwargs and hasattr(self, '_module_inputs'):
            kwargs['inputs'] = self._module_inputs
        if 'forward_fn' not in kwargs and hasattr(self, '_module_forward_fn'):
            kwargs['forward_fn'] = self._module_forward_fn
        return self.output.backward(**kwargs)


def extract_and_wrap_module_forward(module: torch.nn.Module, *module_inputs):
    """
    从 PyTorch Module 提取 forward 函数并包装为 MindSpore autograd 函数
    
    这是 torch4ms 版本的 "functional_call + autograd wrapper"
    
    Args:
        module: PyTorch 模块
        *module_inputs: 模块的输入（torch4ms.Tensor 或 torch.Tensor）
    
    Returns:
        ModuleOutputWrapper: 包装对象，包含 output、forward_fn 和 inputs
    """
    # 获取环境
    env = None
    if module_inputs:
        if isinstance(module_inputs[0], Tensor):
            env = module_inputs[0]._env
        else:
            import torch4ms
            env = torch4ms.default_env()
    else:
        import torch4ms
        env = torch4ms.default_env()
    
    # 提取 forward 函数（MindSpore 版本）
    extractor = AutogradForwardExtractor(module)
    
    # 对于 Linear 模块，直接使用 MindSpore 算子实现（避免 functional_call 的问题）
    if isinstance(module, torch.nn.Linear):
        from mindspore import ops
        def linear_forward_fn(weight_ms, bias_ms, x_ms):
            """直接使用 MindSpore 算子实现 Linear"""
            return ops.matmul(x_ms, ops.transpose(weight_ms, (1, 0))) + bias_ms
        forward_fn = linear_forward_fn
    else:
        forward_fn = extractor.get_ms_forward_fn(env, include_buffers=False)
    
    # 获取参数
    params = extractor.get_trainable_params()
    
    # 转换参数为 torch4ms.Tensor（用于 autograd）
    param_tensors = []
    for param in params:
        if isinstance(param, Tensor):
            param_tensors.append(param)
        elif isinstance(param, torch.Tensor):
            # 转换为 MindSpore 张量
            from torch4ms.ops import mappings
            ms_param = mappings.t2ms(param)
            param_tensors.append(Tensor(ms_param, env, requires_grad=True))
        else:
            raise TypeError(f"Unsupported parameter type: {type(param)}")
    
    # 转换输入为 torch4ms.Tensor（如果需要）
    input_tensors = []
    for inp in module_inputs:
        if isinstance(inp, Tensor):
            input_tensors.append(inp)
        elif isinstance(inp, torch.Tensor):
            ms_inp = env.t2ms_copy(inp)
            input_tensors.append(env.ms2t_iso(ms_inp))
        else:
            input_tensors.append(inp)
    
    # 将所有参数和输入合并
    all_args = tuple(param_tensors) + tuple(input_tensors)
    
    # 创建一个只接受 params 的 forward_fn（inputs 作为闭包变量）
    # 这样 backward 时只需要传入 params
    input_tensors_ms = tuple(inp._elem if isinstance(inp, Tensor) else inp for inp in input_tensors)
    
    def params_only_forward_fn(*ms_params):
        """只接受 params 的 forward_fn，inputs 作为闭包变量"""
        return forward_fn(*ms_params, *input_tensors_ms)
    
    # 包装为 autograd 函数（只传入 params）
    output = ms2t_autograd(params_only_forward_fn, *param_tensors)
    
    # 返回包装对象，保存 forward_fn 和 inputs 信息
    return ModuleOutputWrapper(output, params_only_forward_fn, tuple(param_tensors))


def extract_and_wrap_loss_fn(module: torch.nn.Module, loss_fn: Callable, *module_inputs):
    """
    提取完整的 loss 函数（包含模型前向和 loss 计算），类似 torchax 的方式
    
    这是 torch4ms 版本的完整 loss 函数包装，确保所有操作（包括 loss_fn 中的 sum, mean 等）
    都能被 GradOperation 正确跟踪。
    
    Args:
        module: PyTorch 模块
        loss_fn: PyTorch 的 loss 函数（如 torch.nn.CrossEntropyLoss()）
                签名：Callable[output, label] -> loss
        *module_inputs: 模块的输入和标签 (inputs, labels)
    
    Returns:
        ModuleOutputWrapper: 包装对象，包含 loss、forward_fn 和 inputs
    """
    # 分离 inputs 和 labels
    if len(module_inputs) < 2:
        raise ValueError("module_inputs should contain at least (inputs, labels)")
    
    inputs = module_inputs[0]
    labels = module_inputs[1] if len(module_inputs) > 1 else None
    
    # 获取环境
    env = None
    if isinstance(inputs, Tensor):
        env = inputs._env
    else:
        import torch4ms
        env = torch4ms.default_env()
    
    # 提取模型前向函数（MindSpore 版本）
    extractor = AutogradForwardExtractor(module)
    
    # 转换输入为 MindSpore Tensor（需要在 model_forward_fn 定义之前）
    input_tensors = []
    if isinstance(inputs, Tensor):
        input_tensors.append(inputs)
    elif isinstance(inputs, torch.Tensor):
        ms_inp = env.t2ms_copy(inputs)
        input_tensors.append(env.ms2t_iso(ms_inp))
    else:
        input_tensors.append(inputs)
    
    # 转换标签为 MindSpore Tensor（如果需要）
    label_tensors = []
    if labels is not None:
        if isinstance(labels, Tensor):
            label_tensors.append(labels)
        elif isinstance(labels, torch.Tensor):
            ms_label = env.t2ms_copy(labels)
            label_tensors.append(env.ms2t_iso(ms_label))
        else:
            label_tensors.append(labels)
    
    # 获取 MindSpore 版本的输入和标签
    input_tensors_ms = tuple(inp._elem if isinstance(inp, Tensor) else inp for inp in input_tensors)
    label_tensors_ms = tuple(label._elem if isinstance(label, Tensor) else label for label in label_tensors) if label_tensors else None
    
    # 对于 Linear 模块，直接使用 MindSpore 算子实现（避免转换问题）
    if isinstance(module, torch.nn.Linear):
        from mindspore import ops
        
        # 获取参数
        params = extractor.get_trainable_params()
        param_tensors = []
        for param in params:
            if isinstance(param, Tensor):
                param_tensors.append(param)
            elif isinstance(param, torch.Tensor):
                from torch4ms.ops import mappings
                ms_param = mappings.t2ms(param)
                param_tensors.append(Tensor(ms_param, env, requires_grad=True))
            else:
                raise TypeError(f"Unsupported parameter type: {type(param)}")
        
        # 创建纯 MindSpore 语义的 forward 函数
        def model_forward_fn(*ms_params):
            """纯 MindSpore 语义的 Linear forward"""
            weight_ms = ms_params[0]
            bias_ms = ms_params[1] if len(ms_params) > 1 else None
            # input_tensors_ms 是闭包变量
            x_ms = input_tensors_ms[0]
            output_ms = ops.matmul(x_ms, ops.transpose(weight_ms, (1, 0)))
            if bias_ms is not None:
                output_ms = output_ms + bias_ms
            return output_ms
    else:
        # 其他模块使用 extractor 的方法
        model_forward_fn = extractor.get_ms_forward_fn(env, include_buffers=False)
        # 获取参数
        params = extractor.get_trainable_params()
        
        # 转换参数为 torch4ms.Tensor（用于 autograd）
        param_tensors = []
        for param in params:
            if isinstance(param, Tensor):
                param_tensors.append(param)
            elif isinstance(param, torch.Tensor):
                from torch4ms.ops import mappings
                ms_param = mappings.t2ms(param)
                param_tensors.append(Tensor(ms_param, env, requires_grad=True))
            else:
                raise TypeError(f"Unsupported parameter type: {type(param)}")
    
    # 创建完整的 loss 函数（MindSpore 语义）
    # 这个函数包含：模型前向 + loss 计算
    # 注意：我们需要将 loss_fn 的操作也转换为 MindSpore 语义
    def full_loss_fn(*ms_params):
        """完整的 loss 函数：模型前向 + loss 计算"""
        # 模型前向
        output_ms = model_forward_fn(*ms_params, *input_tensors_ms)
        
        # 调用 loss 函数（直接使用 MindSpore Tensor）
        # 假设 loss_fn 是 MindSpore 语义的函数
        if label_tensors_ms:
            loss_ms = loss_fn(output_ms, label_tensors_ms[0])
        else:
            loss_ms = loss_fn(output_ms)
        
        
        # 确保返回 MindSpore Tensor
        if isinstance(loss_ms, ms_Tensor):
            return loss_ms
        else:
            # 如果不是 MindSpore Tensor，尝试转换
            try:
                return ms_Tensor(loss_ms)
            except:
                return loss_ms
        
        # 转换回 MindSpore Tensor
        # 注意：loss_t4 可能已经是 MindSpore Tensor（如果 loss_fn 被正确拦截）
        if isinstance(loss_t4, Tensor):
            return loss_t4._elem
        elif isinstance(loss_t4, torch.Tensor):
            from torch4ms.ops import mappings
            return mappings.t2ms(loss_t4)
        elif isinstance(loss_t4, ms_Tensor):
            return loss_t4
        else:
            # 尝试转换为 MindSpore Tensor
            try:
                return ms_Tensor(loss_t4)
            except:
                return loss_t4
    
    # 包装为 autograd 函数（只传入 params）
    loss_output = ms2t_autograd(full_loss_fn, *param_tensors)
    
    # 保存 forward_fn 和 inputs 到 loss Tensor（用于 backward）
    loss_output._module_forward_fn = full_loss_fn
    loss_output._module_inputs = tuple(param_tensors)
    
    # 返回包装对象
    return ModuleOutputWrapper(loss_output, full_loss_fn, tuple(param_tensors))


def make_train_step(model_fn: Callable, loss_fn: Callable, optimizer=None):
    """
    创建训练步骤函数，类似 torchax 的 make_train_step
    
    关键思路：将 model_fn 和 loss_fn 组合成完整的 loss 函数，
    然后使用 GradOperation 对整个函数求梯度，自动跟踪所有操作。
    
    Args:
        model_fn: 函数式的模型前向函数
            Callable[weights, buffers, args] -> result
            或者直接是 torch.nn.Module
        loss_fn: PyTorch 的 loss 函数（如 torch.nn.CrossEntropyLoss()）
            Callable[result, label] -> loss
        optimizer: 优化器（可选）
    
    Returns:
        step 函数：执行一个训练步骤
            Callable[weights, buffers, args, labels] -> (loss, weights, buffers)
    """
    import torch4ms
    from torch4ms.autograd.forward_extractor import AutogradForwardExtractor
    
    env = torch4ms.default_env()
    
    # 如果 model_fn 是 torch.nn.Module，提取函数式版本
    if isinstance(model_fn, torch.nn.Module):
        module = model_fn
        extractor = AutogradForwardExtractor(module)
        
        def step(inputs, labels):
            """执行一个训练步骤（使用 Module）"""
            with env:
                # 使用 extract_and_wrap_loss_fn 提取完整的 loss 函数
                loss_wrapper = extract_and_wrap_loss_fn(module, loss_fn, inputs, labels)
                loss = loss_wrapper.output
                
                # backward（自动使用保存的 forward_fn）
                loss.backward()
                
                # 更新参数（如果有优化器）
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
                
                return loss, module
    else:
        # model_fn 是函数式的
        def step(weights, buffers, args, labels):
            """执行一个训练步骤（使用函数式 model_fn）"""
            with env:
                # 模型前向
                output = model_fn(weights, buffers, args)
                
                # loss 计算（PyTorch 的 loss 函数，会被 torch4ms 拦截）
                loss = loss_fn(output, labels)
                
                # backward（需要提取完整的 forward_fn）
                # 这里需要将整个 (model_fn + loss_fn) 组合成 forward_fn
                # 然后使用 GradOperation 求梯度
                loss.backward()
                
                # 更新参数（如果有优化器）
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
                
                return loss, weights, buffers
    
    return step
