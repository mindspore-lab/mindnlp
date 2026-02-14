"""autograd module for torch4ms"""
import warnings
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, Union
import numpy as np
from mindspore import Tensor as ms_Tensor
from mindspore import Parameter
from mindspore import ops
from mindspore.ops import GradOperation
import mindspore
import torch4ms
import torch
from torch4ms.tensor import Tensor
from torch4ms.autograd.graph_tracker import get_computation_graph
from torch4ms.autograd.forward_extractor import AutogradForwardExtractor


def _to_torch4ms_tensor(value, env) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, ms_Tensor):
        return env.ms2t_iso(value)
    if isinstance(value, torch.Tensor):
        return env.ms2t_iso(env.t2ms_copy(value))
    raise TypeError(f"Unsupported parameter type: {type(value).__name__}")


def _to_ms_value(value, env):
    if isinstance(value, Tensor):
        return value._elem
    if isinstance(value, ms_Tensor):
        return value
    if isinstance(value, torch.Tensor):
        return env.t2ms_copy(value)
    return value

_OptionalTensor = Optional[Tensor]
_ShapeorNestedShape = Union[Tuple[int, ...], Sequence[Tuple[int, ...]], Tensor]


def _is_checkpoint_valid():
    return True


def _calculate_shape(
    output: Tensor, grad: Tensor, is_grads_batched: bool
) -> Tuple[_ShapeorNestedShape, _ShapeorNestedShape]:
    """Calculate and compare shapes of output and grad tensors."""
    out_shape = output.shape
    reg_grad_shape = grad.shape if not is_grads_batched else grad.shape[1:]
    return out_shape, reg_grad_shape


def _tensor_or_tensors_to_tuple(
    tensors: Optional[Union[Tensor, Sequence[Tensor]]], length: int
) -> Tuple[_OptionalTensor, ...]:
    """Convert tensor or sequence of tensors to tuple."""
    if tensors is None:
        return (None,) * length
    if isinstance(tensors, Tensor):
        return (tensors,)
    return tuple(tensors)


def _make_grads(
    outputs: Sequence[Tensor],
    grads: Sequence[_OptionalTensor],
    is_grads_batched: bool,
) -> Tuple[_OptionalTensor, ...]:
    """Validate and prepare gradients for backward pass."""
    new_grads: List[_OptionalTensor] = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, Tensor):
            first_grad = grad if not is_grads_batched else grad._elem[0]
            if out.shape != first_grad.shape:
                out_shape, grad_shape = _calculate_shape(out, first_grad, is_grads_batched)
                if is_grads_batched:
                    raise RuntimeError(
                        "If `is_grads_batched=True`, we interpret the first "
                        "dimension of each grad_output as the batch dimension. "
                        "The sizes of the remaining dimensions are expected to match "
                        "the shape of corresponding output, but a mismatch "
                        "was detected: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + ". "
                        "If you only want some tensors in `grad_output` to be considered "
                        "batched, consider using vmap."
                    )
                else:
                    raise RuntimeError(
                        "Mismatch in shape: grad_output["
                        + str(grads.index(grad))
                        + "] has a shape of "
                        + str(grad_shape)
                        + " and output["
                        + str(outputs.index(out))
                        + "] has a shape of "
                        + str(out_shape)
                        + "."
                    )
            new_grads.append(grad)
        elif grad is None:
            if out._elem.size == 0 or out._elem.size > 1:
                # For non-scalar outputs, we need explicit gradients
                # For scalar outputs, we can create ones_like
                if out._elem.size != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
            # Create ones_like for scalar outputs
            ones_grad = ops.ones_like(out._elem)
            new_grads.append(Tensor(ones_grad, out._env, requires_grad=False))
        else:
            raise TypeError("gradients can be either Tensors or None, but got " + type(grad).__name__)
    return tuple(new_grads)


def backward(
    tensors: Union[Tensor, Sequence[Tensor]],
    grad_tensors: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    inputs: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    module_inputs: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    forward_fn: Optional[Callable] = None,
    module: Optional["torch.nn.Module"] = None,
) -> None:
    """
    Computes the sum of gradients of given tensors w.r.t. graph leaves.
    
    Note: In MindSpore, we need to explicitly provide inputs for gradient computation
    using GradOperation, as intermediate results don't have requires_grad or grad_fn.
    
    Args:
        tensors: Output tensors to compute gradients for
        grad_tensors: Gradient w.r.t. each output tensor
        retain_graph: If False, the graph will be freed
        create_graph: If True, graph of the derivative will be constructed
        grad_variables: Deprecated, use inputs instead
        inputs: Inputs w.r.t. which gradients will be accumulated.
                If None and module is provided, will use all trainable parameters of the module.
        forward_fn: Optional explicit forward function. If provided, this will be used
                    instead of rebuilding from computation graph.
        module: Optional torch.nn.Module. If provided, will automatically extract forward_fn
                and inputs from the module. This is the recommended way for module-based training.
        module_inputs: Optional module forward inputs used to build a pure forward function.
                Example: loss.backward(module=model, module_inputs=(x,))
    
    Examples:
        >>> # Method 1: Manual forward function (current way)
        >>> loss.backward(inputs=[x, y], forward_fn=lambda x, y: (x + y).sum())
        
        >>> # Method 2: Automatic extraction from module (recommended)
        >>> model = torch.nn.Linear(10, 1)
        >>> loss = model(x).sum()
        >>> loss.backward(module=model)  # Automatically extracts forward_fn and inputs
    """
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) arguments both passed to backward(). Please only use 'grad_tensors'.")
    
    if inputs is not None and len(inputs) == 0:
        raise RuntimeError("'inputs' argument to backward() cannot be empty.")
    
    # 检查 Tensor 是否有保存的 forward_fn 和 inputs（来自 ModuleOutputWrapper）
    # 先转换为元组以便检查
    tensors_tuple = (tensors,) if isinstance(tensors, Tensor) else tuple(tensors) if tensors else tuple()
    if len(tensors_tuple) > 0 and isinstance(tensors_tuple[0], Tensor):
        first_tensor = tensors_tuple[0]
        if hasattr(first_tensor, '_module_forward_fn') and hasattr(first_tensor, '_module_inputs'):
            if forward_fn is None:
                forward_fn = first_tensor._module_forward_fn
            if inputs is None or len(inputs) == 0:
                inputs = first_tensor._module_inputs
    
    # 如果提供了 module，自动提取 forward_fn 和 inputs
    if module is not None:
        env = tensors_tuple[0]._env if len(tensors_tuple) > 0 else torch4ms.default_env()
        extractor = AutogradForwardExtractor(module)
        if forward_fn is None:
            forward_fn = extractor.get_ms_forward_fn(env, include_buffers=False)
        if inputs is None or len(inputs) == 0:
            # 自动使用所有可训练参数作为 inputs（torch4ms.Tensor）
            param_list = extractor.get_trainable_params()
            inputs = tuple(_to_torch4ms_tensor(param, env) for param in param_list)

        # 将模块前向输入绑定到 forward_fn（避免依赖计算图重建）
        if module_inputs is not None:
            module_inputs_tuple = (module_inputs,) if isinstance(module_inputs, Tensor) else tuple(module_inputs)
            module_inputs_ms = tuple(_to_ms_value(val, env) for val in module_inputs_tuple)

            def bound_forward_fn(*ms_params):
                return forward_fn(*ms_params, *module_inputs_ms)

            forward_fn = bound_forward_fn
    
    # Convert to tuples (使用前面转换的结果)
    tensors = tensors_tuple
    inputs = ((inputs,) if isinstance(inputs, Tensor) else tuple(inputs) if inputs is not None else tuple())
    grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    
    if retain_graph is None:
        retain_graph = create_graph
    
    # Use GradOperation to compute gradients
    # If forward_fn is provided, use it directly; otherwise rebuild from computation graph
    # 保存模块信息用于梯度同步
    _run_backward(tensors, grad_tensors_, retain_graph, create_graph, inputs, forward_fn=forward_fn, module=module)


def _sync_grad_to_param_grad(param_tensor: Tensor, grad_value: Tensor, env, module: Optional["torch.nn.Module"] = None, param_index: Optional[int] = None):
    """
    将 torch4ms.Tensor 的梯度同步到模型参数的 grad 属性
    
    如果提供了 module 和 param_index，将梯度同步到对应的模型参数。
    否则，如果 param_tensor 本身就是 torch.nn.Parameter，直接同步。
    
    Args:
        param_tensor: torch4ms.Tensor（可能是模型参数）
        grad_value: 梯度值（torch4ms.Tensor）
        env: Environment 对象
        module: 可选的 PyTorch 模块，用于查找对应的参数
        param_index: 可选的参数索引，用于在 module 中查找对应的参数
    """
    import torch.utils._mode_utils as mode_utils
    from torch4ms.ops import mappings
    
    try:
        # 如果提供了 module 和 param_index，直接同步到对应的参数
        if module is not None and param_index is not None:
            param_list = list(module.parameters())
            if param_index < len(param_list):
                original_param = param_list[param_index]
                # 将梯度转换为普通 torch.Tensor 并设置到 grad 属性
                with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                    grad_torch = mappings.ms2t(grad_value._elem)
                with torch.no_grad():
                    original_param.grad = grad_torch
                return
        
        # 如果 param_tensor 本身就是 torch.nn.Parameter（继承自 Tensor）
        if isinstance(param_tensor, torch.nn.Parameter):
            # 将梯度转换为普通 torch.Tensor 并设置到 grad 属性
            with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                grad_torch = mappings.ms2t(grad_value._elem)
            with torch.no_grad():
                param_tensor.grad = grad_torch
    except Exception:
        # 如果同步失败，忽略错误（不影响 _t4ms_grad 的设置）
        pass


def _run_backward(
    tensors: Tuple[Tensor, ...],
    grad_tensors: Tuple[Optional[Tensor], ...],
    retain_graph: bool,
    create_graph: bool,
    inputs: Tuple[Tensor, ...],
    forward_fn: Optional[Callable] = None,
    module: Optional["torch.nn.Module"] = None,
) -> None:
    """
    Internal function to run backward pass using MindSpore.
    
    使用路线2：通过 GradOperation 计算梯度。
    根据计算图跟踪器重建 forward 函数。
    """
    if len(tensors) == 0:
        return
    
    # 如果指定了 inputs，转换为 MindSpore tensors
    if len(inputs) == 0:
        warnings.warn("No inputs specified for backward. In MindSpore, you need to explicitly "
                    "provide inputs parameter when calling backward(), as intermediate results "
                    "don't have requires_grad or grad_fn. Example: loss.backward(inputs=[x, y])")
        return
    
    ms_inputs = tuple(inp._elem for inp in inputs)
    ms_outputs = tuple(t._elem for t in tensors)
    ms_grads = tuple(g._elem if g is not None else ops.ones_like(t._elem) for g, t in zip(grad_tensors, tensors))
    
    try:
        # 使用计算图跟踪器重建 forward 函数
        graph = get_computation_graph()
        
        if len(tensors) == 1:
            output_tensor = tensors[0]
            grad_output = ms_grads[0]
            
            # 如果提供了显式的 forward 函数，使用它；否则从计算图重建
            if forward_fn is None:
                forward_fn = _rebuild_forward_fn(output_tensor, inputs, graph)
            
            if forward_fn is None:
                warnings.warn("Could not rebuild forward function from computation graph. "
                            "This may happen if the computation involves operations not tracked. "
                            "Consider providing an explicit forward function: "
                            "loss.backward(inputs=[x, y], forward_fn=lambda x, y: (x + y).sum())")
                return
            
            # 使用 GradOperation 计算梯度
            # 根据 MindSpore API: GradOperation(get_all=True, sens_param=True)
            grad_op = GradOperation(get_all=True, sens_param=True)
            grad_fn = grad_op(forward_fn)
            
            try:
                # 调用 grad_fn: grad_fn(*inputs, grad_output)
                grads = grad_fn(*ms_inputs, grad_output)

                # Optional debug (enabled via env var TORCH4MS_DEBUG_BACKWARD=1)
                import os
                if os.environ.get("TORCH4MS_DEBUG_BACKWARD") == "1":
                    try:
                        print("[torch4ms][backward] inputs:", len(inputs))
                        print("[torch4ms][backward] inputs.requires_grad:", [getattr(i, "requires_grad", None) for i in inputs])
                        print("[torch4ms][backward] ms_inputs types:", [type(x).__name__ for x in ms_inputs])
                        print("[torch4ms][backward] grads type:", type(grads).__name__)
                        if hasattr(grads, "__len__"):
                            print("[torch4ms][backward] grads len:", len(grads))
                        # show None-ness
                        try:
                            print("[torch4ms][backward] grads is None list:", [g is None for g in grads])
                        except Exception:
                            pass
                    except Exception:
                        pass
                
                # 累积梯度到 .grad 属性
                env = tensors[0]._env
                # IMPORTANT: torch.Tensor wrapper subclasses may route even attribute assignment
                # (e.g. setting .grad) through TorchFunctionMode as "__set__" while env is enabled.
                # Guard gradient writes to avoid interception / silent failures.
                import os as _os
                _debug = _os.environ.get("TORCH4MS_DEBUG_BACKWARD") == "1"
                with torch._C.DisableTorchFunction():
                    for i, inp in enumerate(inputs):
                        if inp.requires_grad and i < len(grads) and grads[i] is not None:
                            grad_value = Tensor(grads[i], env, requires_grad=False)
                            if _debug:
                                try:
                                    print("[torch4ms][backward] write grad for input", i, "before:", object.__getattribute__(inp, "_t4ms_grad"))
                                except Exception:
                                    pass
                            existing = None
                            try:
                                existing = object.__getattribute__(inp, "_t4ms_grad")
                            except Exception:
                                existing = None

                            if existing is not None:
                                # 累积梯度（torch4ms-side）
                                try:
                                    object.__setattr__(
                                        inp,
                                        "_t4ms_grad",
                                        Tensor(existing._elem + grads[i], env, requires_grad=False),
                                    )
                                except Exception:
                                    # Fallback: overwrite if accumulation fails
                                    object.__setattr__(inp, "_t4ms_grad", grad_value)
                            else:
                                # 设置梯度（torch4ms-side）
                                object.__setattr__(inp, "_t4ms_grad", grad_value)
                            
                            # 自动同步梯度到模型参数的 grad 属性（优先级 2：梯度自动同步）
                            # 如果提供了 module，尝试同步到对应的模型参数
                            # 这样 PyTorch 优化器就可以直接使用 param.grad
                            _sync_grad_to_param_grad(inp, grad_value, env, module=module, param_index=i)
                            
                            if _debug:
                                try:
                                    print("[torch4ms][backward] write grad for input", i, "after:", object.__getattribute__(inp, "_t4ms_grad"))
                                except Exception:
                                    pass
            except Exception as e:
                warnings.warn(f"GradOperation failed in backward: {e}. "
                            "This may indicate that the forward function could not be properly rebuilt. "
                            "Consider using grad() function with explicit forward function.")
        else:
            warnings.warn("Multiple outputs in backward not fully supported yet. "
                        "Only single output backward is currently supported.")
    except Exception as e:
        warnings.warn(f"Error in backward pass: {e}. This may indicate that the computation graph "
                     "was not properly recorded. Ensure operations are performed on tensors with "
                     "requires_grad=True.")


def _rebuild_forward_fn(output: Tensor, inputs: List[Tensor], graph) -> Optional[Callable]:
    """
    根据计算图跟踪器重建 forward 函数
    
    支持操作链的重建，例如：z = x + y; loss = z.sum()
    
    Args:
        output: 输出 Tensor
        inputs: 输入 Tensor 列表
        graph: 计算图跟踪器
        
    Returns:
        forward 函数，如果无法重建则返回 None
    """
    output_id = id(output)
    
    # 检查输出是否在计算图中
    if output_id not in graph.output_to_op:
        return None
    
    record = graph.output_to_op[output_id]

    def _resolve_inputs(raw_inputs):
        resolved = []
        for inp in raw_inputs:
            if hasattr(inp, '__call__'):
                try:
                    inp = inp()
                except Exception:
                    inp = None
            resolved.append(inp)
        return resolved

    input_values = _resolve_inputs(record.inputs)
    
    # 对于 sum 操作，需要特殊处理操作链
    if record.op_name == 'sum':
        # sum 的输入可能是中间结果
        if len(input_values) > 0:
            sum_input = input_values[0]
            # 检查 sum_input 是否是 inputs 中的一个（直接输入）
            # 使用 is 操作符进行对象标识比较，而不是 in（因为 Tensor.__eq__ 是逐元素比较）
            idx = None
            for i, inp in enumerate(inputs):
                if inp is sum_input:
                    idx = i
                    break
            if idx is not None:
                # sum 的输入就是 target_inputs 中的一个
                def forward_fn(*args):
                    return ops.reduce_sum(args[idx])
                return forward_fn
            else:
                # sum 的输入是中间结果，需要递归重建
                sum_input_id = id(sum_input)
                if sum_input_id in graph.output_to_op:
                    # 递归重建 sum_input 的 forward 函数
                    sum_forward_fn = _rebuild_forward_fn(sum_input, inputs, graph)
                    if sum_forward_fn is not None:
                        def forward_fn(*args):
                            intermediate = sum_forward_fn(*args)
                            return ops.reduce_sum(intermediate)
                        return forward_fn
                # 如果递归重建失败，返回 None（继续尝试其他处理方式）
                # 但 sum 操作无法匹配其他操作类型，所以这里返回 None 是合理的
    
    # 对于简单的二元运算
    if record.op_name in ['add', 'sub', 'mul', 'div']:
        # 检查输入是否匹配
        if len(input_values) == 2:
            input0 = input_values[0]
            input1 = input_values[1]
            
            # 检查两个输入是否在 inputs 中（直接输入）
            idx0 = None
            idx1 = None
            for i, inp in enumerate(inputs):
                if inp is input0:
                    idx0 = i
                if inp is input1:
                    idx1 = i
            
            # 情况1：两个输入都是 inputs 中的直接 Tensor
            if idx0 is not None and idx1 is not None:
                def forward_fn(*args):
                    return record.op_func(args[idx0], args[idx1])
                return forward_fn
            
            # 情况2：两个输入都不是 inputs 中的直接 Tensor（都是中间结果或常量）
            elif idx0 is None and idx1 is None:
                fn0 = _rebuild_forward_fn(input0, inputs, graph) if isinstance(input0, Tensor) else None
                fn1 = _rebuild_forward_fn(input1, inputs, graph) if isinstance(input1, Tensor) else None
                if fn0 is not None and fn1 is not None:
                    def forward_fn(*args):
                        val0 = fn0(*args)
                        val1 = fn1(*args)
                        return record.op_func(val0, val1)
                    return forward_fn
            
            # 情况3：只有一个输入是 inputs 中的直接 Tensor
            elif idx0 is not None or idx1 is not None:
                if idx0 is not None:
                    other = input1
                    if isinstance(other, Tensor):
                        other_fn = _rebuild_forward_fn(other, inputs, graph)
                        if other_fn is not None:
                            def forward_fn(*args):
                                other_val = other_fn(*args)
                                return record.op_func(args[idx0], other_val)
                            return forward_fn
                        def forward_fn(*args):
                            return record.op_func(args[idx0], other._elem)
                        return forward_fn
                    def forward_fn(*args):
                        return record.op_func(args[idx0], other)
                    return forward_fn

                if idx1 is not None:
                    other = input0
                    if isinstance(other, Tensor):
                        other_fn = _rebuild_forward_fn(other, inputs, graph)
                        if other_fn is not None:
                            def forward_fn(*args):
                                other_val = other_fn(*args)
                                return record.op_func(other_val, args[idx1])
                            return forward_fn
                        def forward_fn(*args):
                            return record.op_func(other._elem, args[idx1])
                        return forward_fn
                    def forward_fn(*args):
                        return record.op_func(other, args[idx1])
                    return forward_fn
    
    # 如果无法重建，返回 None
    return None


def grad(
    outputs: Union[Tensor, Sequence[Tensor]],
    inputs: Union[Tensor, Sequence[Tensor]],
    grad_outputs: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: Optional[bool] = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
    forward_fn: Optional[Callable] = None,
    module: Optional["torch.nn.Module"] = None,
    module_inputs: Optional[Union[Tensor, Sequence[Tensor]]] = None,
) -> Tuple[Tensor, ...]:
    """
    Computes and returns the sum of gradients of outputs w.r.t. the inputs.
    
    Args:
        outputs: Outputs of the differentiated function.
        inputs: Inputs w.r.t. which the gradient will be returned.
                If None and module is provided, will use all trainable parameters of the module.
        grad_outputs: The "vector" in the Jacobian-vector product. If None, defaults to ones.
        retain_graph: If False, the graph used to compute the grads will be freed.
        create_graph: If True, graph of the derivative will be constructed.
        only_inputs: If True, only gradients w.r.t. inputs are returned.
        allow_unused: If False, specifying inputs that were not used will raise an error.
        is_grads_batched: If True, grad_outputs are interpreted as batched.
        materialize_grads: If True, unused gradients are materialized as zeros.
        forward_fn: Optional explicit forward function. If provided, this will be used
                    instead of rebuilding from computation graph.
        module: Optional torch.nn.Module. If provided, will automatically extract forward_fn
                and inputs from the module.
        module_inputs: Optional module forward inputs used to build a pure forward function.
    
    Returns:
        Tuple of gradients w.r.t. inputs.
    
    Examples:
        >>> # Method 1: Manual forward function
        >>> x_grad, y_grad = grad(loss, [x, y], forward_fn=lambda x, y: (x + y).sum())
        
        >>> # Method 2: Automatic extraction from module
        >>> model = torch.nn.Linear(10, 1)
        >>> loss = model(x).sum()
        >>> grads = grad(loss, module=model)  # Automatically extracts forward_fn and inputs
    """
    # 如果提供了 module，自动提取 forward_fn 和 inputs
    if module is not None:
        env = outputs[0]._env if isinstance(outputs, Tensor) else (outputs[0]._env if len(outputs) > 0 else torch4ms.default_env())
        extractor = AutogradForwardExtractor(module)
        if forward_fn is None:
            forward_fn = extractor.get_ms_forward_fn(env, include_buffers=False)
        if inputs is None:
            # 自动使用所有可训练参数作为 inputs
            param_list = extractor.get_trainable_params()
            inputs = tuple(_to_torch4ms_tensor(param, env) for param in param_list)

        # 将模块前向输入绑定到 forward_fn（避免依赖计算图重建）
        if module_inputs is not None:
            module_inputs_tuple = (module_inputs,) if isinstance(module_inputs, Tensor) else tuple(module_inputs)
            module_inputs_ms = tuple(_to_ms_value(val, env) for val in module_inputs_tuple)

            def bound_forward_fn(*ms_params):
                return forward_fn(*ms_params, *module_inputs_ms)

            forward_fn = bound_forward_fn
    
    if inputs is None or len(inputs) == 0:
        raise ValueError("grad requires non-empty inputs. Either provide inputs explicitly or provide module parameter.")
    
    assert not is_grads_batched, "The argument is_grads_batched must be False now due to the lack of support for vmap in mindspore!"
    
    if materialize_grads and allow_unused is False:
        raise ValueError("Expected allow_unused to be True or not passed when materialize_grads=True, but got: allow_unused=False.")
    
    if allow_unused is None:
        allow_unused = materialize_grads
    
    t_outputs = cast(Tuple[Tensor, ...], (outputs,) if isinstance(outputs, Tensor) else tuple(outputs))
    t_inputs = cast(Tuple[Tensor, ...], (inputs,) if isinstance(inputs, Tensor) else tuple(inputs))
    
    if not only_inputs:
        warnings.warn("only_inputs argument is deprecated and is ignored now (defaults to True). To accumulate gradient for other "
            "parts of the graph, please use torch4ms.autograd.backward.")
    
    grad_outputs_ = _tensor_or_tensors_to_tuple(grad_outputs, len(t_outputs))
    grad_outputs_ = _make_grads(t_outputs, grad_outputs_, is_grads_batched=is_grads_batched)
    
    if retain_graph is None:
        retain_graph = create_graph
    
    # Compute gradients using MindSpore
    result = _run_grad(t_outputs, grad_outputs_, retain_graph, create_graph, t_inputs, allow_unused, forward_fn=forward_fn)
    
    if materialize_grads:
        result = tuple(
            output if output is not None else Tensor(ops.zeros_like(inp._elem), inp._env, requires_grad=True)
            for (output, inp) in zip(result, t_inputs)
        )
    
    return result


def _run_grad(
    outputs: Tuple[Tensor, ...],
    grad_outputs: Tuple[Optional[Tensor], ...],
    retain_graph: bool,
    create_graph: bool,
    inputs: Tuple[Tensor, ...],
    allow_unused: bool,
    forward_fn: Optional[Callable] = None,
) -> Tuple[Optional[Tensor], ...]:
    """
    Internal function to compute gradients using MindSpore.
    
    使用路线2：通过 GradOperation 计算梯度。
    根据计算图跟踪器重建 forward 函数。
    """
    if len(outputs) == 0 or len(inputs) == 0:
        return tuple(None for _ in inputs)
    
    env = outputs[0]._env
    
    # Convert to MindSpore tensors
    ms_outputs = tuple(out._elem for out in outputs)
    ms_grads = tuple(g._elem if g is not None else ops.ones_like(out._elem) for g, out in zip(grad_outputs, outputs))
    ms_inputs = tuple(inp._elem for inp in inputs)
    
    try:
        # 使用计算图跟踪器重建 forward 函数
        graph = get_computation_graph()
        
        if len(outputs) == 1:
            output_tensor = outputs[0]
            grad_output = ms_grads[0]
            
            # 如果提供了显式的 forward 函数，使用它；否则从计算图重建
            if forward_fn is None:
                forward_fn = _rebuild_forward_fn(output_tensor, list(inputs), graph)
            
            if forward_fn is None:
                warnings.warn("Could not rebuild forward function from computation graph. "
                            "This may happen if the computation involves operations not tracked. "
                            "You may need to provide an explicit forward function: "
                            "grad(loss, [x, y], forward_fn=lambda x, y: (x + y).sum())")
                return tuple(None for _ in inputs)
            
            # 使用 GradOperation 计算梯度
            # 根据 MindSpore API: GradOperation(get_all=True, sens_param=True)
            grad_op = GradOperation(get_all=True, sens_param=True)
            grad_fn = grad_op(forward_fn)
            
            try:
                # 调用 grad_fn: grad_fn(*inputs, grad_output)
                grads = grad_fn(*ms_inputs, grad_output)
                
                # 将结果转换为 torch4ms Tensor
                results = []
                for i, (grad, inp) in enumerate(zip(grads, inputs)):
                    if grad is not None:
                        results.append(Tensor(grad, env, requires_grad=False))
                    elif allow_unused:
                        results.append(None)
                    else:
                        raise RuntimeError(f"Gradient for input {i} is None and allow_unused=False")
                return tuple(results)
            except Exception as e:
                warnings.warn(f"GradOperation failed in grad: {e}. "
                            "This may indicate that the forward function could not be properly rebuilt.")
                return tuple(None for _ in inputs)
        else:
            warnings.warn("Multiple outputs in grad not fully supported yet. "
                        "Only single output grad is currently supported.")
            return tuple(None for _ in inputs)
    except Exception as e:
        warnings.warn(f"Error in grad computation: {e}. This may indicate that the computation graph "
                     "was not properly recorded. Ensure operations are performed on tensors with "
                     "requires_grad=True.")
        return tuple(None for _ in inputs)

