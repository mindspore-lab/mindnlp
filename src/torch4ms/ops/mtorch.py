"""Tensor constructor overrides"""

import math
import collections.abc
import functools
from typing import Optional, Sequence, Tuple
import numpy as np

import mindspore as ms
import mindspore.ops as mops
import mindspore.numpy as mnp

import torch
from torch4ms.ops.ops_registry import register_torch_function_op
from torch4ms.ops import op_base, mappings
import torch4ms.tensor
from torch4ms.view import View, NarrowInfo
import torch.utils._pytree as pytree
from torch4ms.ops.maten import _aten_convolution


def register_function(torch_func, **kwargs):
    return functools.partial(register_torch_function_op, torch_func, **kwargs)


@register_function(torch.as_tensor, is_mindspore_function=False, needs_env=True)
@op_base.convert_dtype(
    use_default_dtype=False
)  # Attempt to infer type from elements
def _as_tensor(data, dtype=None, device=None, env=None):
    if isinstance(data, torch.Tensor):
        return env._to_copy(data, dtype, device)
    if isinstance(data, np.ndarray):
        ms_res = mnp.asarray(data)
    else:
        ms_res = _tensor(data, dtype=dtype)
    return torch4ms.tensor.Tensor(ms_res, env)


@register_function(torch.tensor)
@op_base.convert_dtype(
    use_default_dtype=False
)  # Attempt to infer type from elements
def _tensor(data, *, dtype=None, **kwargs):
    python_types_to_torch_types = {
        bool: mnp.bool_,
        int: mnp.int64,
        float: mnp.float32,
        complex: ms.complex64,
    }
    if not dtype:
        # MindSpore不直接支持tree_leaves，使用numpy替代
        if isinstance(data, (list, tuple)) and data:
            dtype = python_types_to_torch_types.get(type(data[0]))

    return mnp.array(
        data, dtype=dtype or mappings.t2ms_dtype(torch.get_default_dtype())
    )


@register_function(torch.allclose)
def _aten_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return mnp.allclose(input, other, rtol, atol, equal_nan)


@register_function(torch.angle)
def _torch_angle(input):
    if input.dtype.name == "int64":
        input = input.astype(mnp.dtype("float32"))
    return ms.ops.angle(input)


@register_function(torch.argsort)
def _torch_argsort(input, dim=-1, descending=False, stable=False):
    expanded = False
    if input.ndim == 0:
        expanded = True
        input = mnp.expand_dims(input, 0)
    res = ms.ops.argsort(input, axis=dim, descending=descending)
    # MindSpore的argsort目前不支持stable参数
    if expanded:
        res = res.squeeze()
    return res


@register_function(torch.diag)
def _diag(input, diagonal=0):
    return mnp.diag(input, k=diagonal)


@register_function(torch.einsum)
@register_function(torch.ops.aten.einsum)
def _einsum(equation, *operands):
    def get_params(*a):
        inner_list = a[0]
        if not isinstance(inner_list, ms.Tensor):
            if len(inner_list) == 1:
                A = inner_list
                return A
            elif len(inner_list) == 2:
                A, B = inner_list
                return A, B
        return operands

    assert isinstance(equation, str), "Only accept str equation"
    filtered_operands = get_params(*operands)
    return ms.ops.einsum(equation, *filtered_operands)


def _sdpa_reference(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    # 使用MindSpore实现的SDPA参考版本
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    
    # 计算注意力权重
    attn_weight = mnp.matmul(query, mnp.transpose(key, axes=(-2, -1))) * scale_factor
    
    # 应用因果掩码
    if is_causal:
        L, S = query.shape[-2], key.shape[-2]
        causal_mask = mnp.tril(mnp.ones((L, S)))
        attn_weight = mnp.where(causal_mask, attn_weight, -mnp.inf)
    
    # 应用注意力掩码
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    
    # GQA处理
    if enable_gqa:
        key_repeats = query.shape[-3] // key.shape[-3]
        value_repeats = query.shape[-3] // value.shape[-3]
        if key_repeats > 1:
            key = mops.tile(key, (1, key_repeats, 1, 1))
        if value_repeats > 1:
            value = mops.tile(value, (1, value_repeats, 1, 1))
    
    # softmax和dropout
    attn_weight = mops.softmax(attn_weight, axis=-1)
    if dropout_p > 0:
        attn_weight = mops.dropout(attn_weight, p=dropout_p)
    
    # 计算输出
    return mnp.matmul(attn_weight, value)


# 移除JAX特定的TPU flash attention实现，MindSpore使用自己的实现方式
# 在scaled_dot_product_attention中我们将直接使用_reference_sdpa


@register_function(torch.nn.functional.one_hot)
def one_hot(tensor, num_classes=-1):
    if num_classes == -1:
        num_classes = mnp.max(tensor) + 1
    return mops.one_hot(tensor, num_classes, on_value=1, off_value=0).astype(mnp.int64)


@register_function(torch.nn.functional.pad)
def pad(tensor, pad, mode="constant", value=None):
    # MindSpore的pad接口需要不同的参数格式
    # 将PyTorch的pad格式转换为MindSpore格式
    # PyTorch: pad = (padding_left, padding_right, padding_top, padding_bottom, ...)
    # MindSpore: pad = ((before_1, after_1), (before_2, after_2), ...)
    
    # 处理负padding值（切片）
    processed_pad = []
    for i in range(0, len(pad), 2):
        pad_left, pad_right = pad[i], pad[i+1]
        processed_pad.append((pad_left, pad_right))
    
    # 反转顺序，因为MindSpore的维度顺序与PyTorch不同
    processed_pad = processed_pad[::-1]
    
    # 模式映射
    mode_map = {
        "constant": "CONSTANT",
        "reflect": "REFLECT",
        "replicate": "SYMMETRIC",
        "circular": "CIRCULAR"
    }
    
    ms_mode = mode_map.get(mode.lower(), "CONSTANT")
    
    # 设置默认值
    if value is None:
        value = 0
    
    # 使用MindSpore的pad操作
    return mops.pad(tensor, processed_pad, mode=ms_mode, constant_values=value)


@register_function(
    torch.nn.functional.scaled_dot_product_attention,
    is_mindspore_function=False,
    needs_env=True,
)
@register_function(
    torch.ops.aten.scaled_dot_product_attention,
    is_mindspore_function=False,
    needs_env=True,
)
def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    env=None,
) -> torch.Tensor:
    # MindSpore不使用TPU flash attention，直接使用我们的参考实现
    return _sdpa_reference(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


@register_function(
    torch.Tensor.__getitem__,
    is_mindspore_function=False,
    is_view_op=True
)
def getitem(self, indexes):
    if isinstance(indexes, list) and isinstance(indexes[0], int):
        # list of int, i.e. x[[1, 2]] NOT x[1, 2] (the second would be tuple of int)
        indexes = (indexes,)
    elif isinstance(indexes, list):
        indexes = tuple(indexes)

    # 检查 self 的类型并获取环境
    from torch4ms import default_env
    
    # 优先检查是否有 _env 和 _elem 属性（torch4ms.Tensor 的特征）
    # 这必须在 isinstance 检查之前，因为可能存在类型检查的问题
    if hasattr(self, '_env') and hasattr(self, '_elem'):
        # 很可能是 torch4ms.Tensor
        try:
            if isinstance(self, torch4ms.tensor.Tensor):
                env = self._env
                elem = self._elem
                parent = self  # torch4ms.Tensor 可以直接作为 parent
            else:
                # 有 _env 和 _elem 但不是 torch4ms.Tensor 类型，可能是 View 等
                env = self._env
                elem = self._elem
                if isinstance(elem, ms.Tensor):
                    parent = torch4ms.tensor.Tensor(elem, env)
                else:
                    parent = self
        except:
            # 如果类型检查失败，直接使用属性
            env = self._env
            elem = self._elem
            if isinstance(elem, ms.Tensor):
                parent = torch4ms.tensor.Tensor(elem, env)
            else:
                parent = self
    # 检查是否是 torch4ms.Tensor（类型检查）
    elif isinstance(self, torch4ms.tensor.Tensor):
        env = self._env
        elem = self._elem
        parent = self  # torch4ms.Tensor 可以直接作为 parent
    # 检查是否是 MindSpore Tensor
    elif isinstance(self, ms.Tensor):
        env = default_env()
        parent = torch4ms.tensor.Tensor(self, env)
        elem = self
    # 检查是否是普通的 PyTorch Tensor（但不是 torch4ms.Tensor）
    # 注意：需要先检查是否有 _env 属性，因为 torch4ms.Tensor 可能有 _env
    elif hasattr(self, '_env') and hasattr(self, '_elem'):
        # 如果有 _env 和 _elem，很可能是 torch4ms.Tensor 但没有被正确识别
        # 或者可能是 View 等其他类型
        env = self._env
        elem = self._elem
        if isinstance(self, torch4ms.tensor.Tensor):
            parent = self
        elif isinstance(elem, ms.Tensor):
            parent = torch4ms.tensor.Tensor(elem, env)
        else:
            # 如果 _elem 不是 ms.Tensor，尝试转换
            try:
                if isinstance(elem, np.ndarray):
                    elem = ms.Tensor(elem)
                else:
                    elem = ms.Tensor(elem)
                parent = torch4ms.tensor.Tensor(elem, env)
            except:
                parent = self
    elif isinstance(self, torch.Tensor):
        # 如果是 PyTorch Tensor（但不是 torch4ms.Tensor），需要获取其底层数据
        env = default_env()
        # 如果 self 有 _elem 属性（可能是 View 或其他包装类型），使用它
        if hasattr(self, '_elem'):
            elem = self._elem
            if isinstance(elem, ms.Tensor):
                parent = torch4ms.tensor.Tensor(elem, env)
            else:
                # 转换为 MindSpore Tensor
                try:
                    # 确保获取的是 numpy 数组，而不是 torch4ms.Tensor
                    if hasattr(self, 'numpy'):
                        np_array = self.numpy()
                        # 如果返回的是 torch4ms.Tensor，使用其 numpy() 方法
                        if isinstance(np_array, torch4ms.tensor.Tensor):
                            np_array = np_array.numpy()
                        elif not isinstance(np_array, np.ndarray):
                            # 如果不是 numpy 数组，尝试转换
                            np_array = np.array(np_array)
                        elem = ms.Tensor(np_array)
                    else:
                        np_array = self.detach().cpu().numpy()
                        elem = ms.Tensor(np_array)
                    parent = torch4ms.tensor.Tensor(elem, env)
                except:
                    # 如果转换失败，尝试使用 _elem
                    if isinstance(elem, (np.ndarray, list)):
                        elem = ms.Tensor(elem)
                        parent = torch4ms.tensor.Tensor(elem, env)
                    else:
                        raise
        else:
            # 直接转换 PyTorch Tensor 到 MindSpore Tensor
            try:
                # 确保获取的是 numpy 数组，而不是 torch4ms.Tensor
                if hasattr(self, 'numpy'):
                    np_array = self.numpy()
                    # 如果返回的是 torch4ms.Tensor，使用其 numpy() 方法
                    if isinstance(np_array, torch4ms.tensor.Tensor):
                        np_array = np_array.numpy()
                    elif not isinstance(np_array, np.ndarray):
                        # 如果不是 numpy 数组，尝试转换
                        np_array = np.array(np_array)
                    elem = ms.Tensor(np_array)
                else:
                    np_array = self.detach().cpu().numpy()
                    elem = ms.Tensor(np_array)
                parent = torch4ms.tensor.Tensor(elem, env)
            except Exception as e:
                # 如果转换失败，可能是 torch4ms.Tensor 但没有被正确识别
                # 尝试检查是否有 _env 属性
                if hasattr(self, '_env'):
                    env = self._env
                    elem = getattr(self, '_elem', None)
                    if elem is not None:
                        parent = torch4ms.tensor.Tensor(elem, env) if not isinstance(self, torch4ms.tensor.Tensor) else self
                    else:
                        raise
                else:
                    raise
    else:
        # 如果是其他类型（可能是 View），尝试获取环境
        if hasattr(self, '_env'):
            env = self._env
            elem = getattr(self, '_elem', None)
            if elem is not None:
                parent = self if hasattr(self, 'ms') else torch4ms.tensor.Tensor(elem, env)
            else:
                parent = self
        elif hasattr(self, 'env'):
            env = self.env
            elem = getattr(self, '_elem', None)
            if elem is not None and isinstance(elem, ms.Tensor):
                parent = self if hasattr(self, 'ms') else torch4ms.tensor.Tensor(elem, env)
            else:
                parent = self
        else:
            # 最后尝试从默认环境获取并转换为 torch4ms.Tensor
            from torch4ms import default_env
            env = default_env()
            if isinstance(self, ms.Tensor):
                parent = torch4ms.tensor.Tensor(self, env)
                elem = self
            else:
                # 尝试获取 _elem 或直接转换
                elem = getattr(self, '_elem', self)
                if isinstance(elem, ms.Tensor):
                    parent = torch4ms.tensor.Tensor(elem, env)
                else:
                    # 如果无法确定，尝试转换
                    try:
                        elem = ms.Tensor(self.numpy() if hasattr(self, 'numpy') else self.detach().cpu().numpy())
                        parent = torch4ms.tensor.Tensor(elem, env)
                    except:
                        parent = self
                        elem = self

    def is_narrow_slicing():
        tensor_free = not pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) or isinstance(x, ms.Tensor),
            indexes,
        )
        list_free = not isinstance(indexes, tuple) or all(
            [False if isinstance(x, list) else True for x in indexes]
        )
        return tensor_free and list_free

    if is_narrow_slicing():
        # 确保 parent 有 ms() 方法，否则转换为 torch4ms.Tensor
        if not hasattr(parent, 'ms') or not callable(getattr(parent, 'ms', None)):
            # 如果 parent 没有 ms() 方法，尝试获取其底层数据并创建 torch4ms.Tensor
            if hasattr(parent, '_elem') and isinstance(parent._elem, ms.Tensor):
                parent = torch4ms.tensor.Tensor(parent._elem, env)
            elif isinstance(parent, torch.Tensor):
                try:
                    if hasattr(parent, 'numpy'):
                        elem = ms.Tensor(parent.numpy())
                    else:
                        elem = ms.Tensor(parent.detach().cpu().numpy())
                    parent = torch4ms.tensor.Tensor(elem, env)
                except:
                    # 如果转换失败，尝试其他方式
                    if hasattr(parent, '_elem'):
                        elem = parent._elem
                        if isinstance(elem, ms.Tensor):
                            parent = torch4ms.tensor.Tensor(elem, env)
                        else:
                            elem = ms.Tensor(elem)
                            parent = torch4ms.tensor.Tensor(elem, env)
                    else:
                        elem = ms.Tensor(parent)
                        parent = torch4ms.tensor.Tensor(elem, env)
        return View(parent, view_info=NarrowInfo(indexes), env=env)

    indexes = env.t2ms_iso(indexes)
    return torch4ms.tensor.Tensor(elem[indexes], env)


@register_function(torch.corrcoef)
def _corrcoef(x):
    if isinstance(x, ms.Tensor) and x.dtype.name == "int64":
        return mnp.corrcoef(x).astype(mnp.float32)
    return mnp.corrcoef(x)


@register_function(torch.sparse.mm, is_mindspore_function=False)
def _sparse_mm(mat1, mat2, reduce="sum"):
    return torch.mm(mat1, mat2)


@register_function(torch.isclose)
def _aten_isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return mnp.isclose(input, other, rtol, atol, equal_nan)


@register_function(torch.linalg.det)
def linalg_det(input):
    return mnp.linalg.det(input)


@register_function(torch.ones)
def _ones(*size: int, dtype=None, **kwargs):
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]
    return mnp.ones(size, dtype=dtype)


@register_function(torch.zeros, is_mindspore_function=False)
def _zeros(*size: int, dtype=None, **kwargs):
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]
    return mnp.zeros(size, dtype=dtype)


@register_function(torch.eye)
@op_base.convert_dtype()
def _eye(n: int, m: Optional[int] = None, *, dtype=None, **kwargs):
    return mnp.eye(n, m, dtype=dtype)


@register_function(torch.full)
@op_base.convert_dtype(use_default_dtype=False)
def _full(size: Sequence[int], fill_value, *, dtype=None, **kwargs):
    # TODO: handle torch.Size
    return mnp.full(size, fill_value, dtype=dtype)


@register_function(torch.empty)
@op_base.convert_dtype()
def empty(*size: Sequence[int], dtype=None, **kwargs):
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]
    return mnp.empty(size, dtype=dtype)


@register_function(torch.arange, is_mindspore_function=False)
def arange(
    start,
    end=None,
    step=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=None,
):
    if end is None:
        end = start
        start = 0
    if step is None:
        step = 1
    return mnp.arange(start, end, step, dtype=dtype)


@register_function(torch.empty_strided, is_mindspore_function=False)
def empty_strided(
    size,
    stride,
    *,
    dtype=None,
    layout=None,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    # MindSpore doesn't support strided tensors directly, using empty as fallback
    return empty(size, dtype=dtype)


@register_function(torch.unravel_index)
def unravel_index(indices, shape):
    return mnp.unravel_index(indices, shape)


@register_function(torch.rand, is_mindspore_function=False, needs_env=True)
def rand(*size, **kwargs):
    if len(size) == 1 and isinstance(size[0], collections.abc.Iterable):
        size = size[0]
    # MindSpore equivalent of uniform random sampling
    return ms.ops.uniform((size,), low=0.0, high=1.0, dtype=kwargs.get('dtype', ms.float32))


@register_function(torch.randn, is_mindspore_function=False, needs_env=True)
def randn(
    *size,
    generator=None,
    out=None,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
    env=None,
):
    # 处理size参数：确保它是一个包含整数的元组
    if not size:
        shape = (1,)
    elif len(size) == 1 and isinstance(size[0], (list, tuple)):
        # 如果传入的是单个列表或元组，如 torch.randn((2, 3))
        shape = tuple(int(x) for x in size[0])
    else:
        # 如果传入的是多个参数，如 torch.randn(2, 3)
        shape = tuple(int(x) for x in size)
    # MindSpore equivalent of normal random sampling
    tensor = ms.ops.normal(shape=shape, mean=0.0, stddev=1.0)
    if dtype is not None:
        # 将PyTorch的dtype转换为MindSpore的dtype
        ms_dtype = mappings.t2ms_dtype(dtype) if isinstance(dtype, torch.dtype) else dtype
        tensor = tensor.astype(ms_dtype)

    # 将 MindSpore Tensor 包装为 torch4ms Tensor，便于后续算子分发与运算
    if env is not None:
        # 使用环境提供的同构转换接口，保持行为一致
        return env.ms2t_iso(tensor)

    # 退化路径：环境缺失时直接返回原始 MindSpore Tensor（保持向后兼容）
    return tensor


@register_function(torch.randint, is_mindspore_function=False, needs_env=True)
def randint(*args, **kwargs):
    # For MindSpore implementation, we'll use uniform and cast to integer
    # This is a simplified implementation
    if len(args) == 1:
        high = args[0]
        low = 0
        size = ()
    elif len(args) == 2:
        low, high = args
        size = ()
    else:
        low, high, size = args[:3]
    
    dtype = kwargs.get('dtype', ms.int32)
    # Generate uniform random values and cast to integer type
    uniform_values = ms.ops.uniform((size,), low=low, high=high, dtype=ms.float32)
    return uniform_values.astype(dtype)


@register_function(torch.logdet)
def logdet(input):
    return ms.ops.logdet(input)


@register_function(torch.linalg.slogdet)
def linalg_slogdet(input):
    # MindSpore equivalent of slogdet
    # Note: MindSpore doesn't have a direct slogdet function, using det and calculating sign
    det_val = mnp.linalg.det(input)
    sign = mnp.sign(det_val)
    logabsdet = mnp.log(mnp.abs(det_val))
    return torch.return_types.slogdet((sign, logabsdet))


@register_function(torch.tensor_split)
def tensor_split(input, indices_or_sections, dim=0):
    return mnp.array_split(input, indices_or_sections, axis=dim)


@register_function(torch.linalg.solve)
def linalg_solve(a, b):
    return mnp.linalg.solve(a, b)


@register_function(torch.linalg.solve_ex)
def linalg_solve_ex(a, b):
    res = mnp.linalg.solve(a, b)
    # For info, return 0 indicating success
    info = ms.Tensor(0, dtype=ms.int32)
    return res, info


@register_function(torch.linalg.svd)
def linalg_svd(a, full_matrices=True):
    return mnp.linalg.svd(a, full_matrices=full_matrices)


@register_function(torch.linalg.matrix_power)
def matrix_power(A, n, *, out=None):
    return mnp.linalg.matrix_power(A, n)


@register_function(torch.svd)
def svd(a, some=True, compute_uv=True):
    if not compute_uv:
        U, S, V = mnp.linalg.svd(a, full_matrices=False)
        # Return empty matrices for U and V as requested
        U = mnp.zeros((a.shape[-2], a.shape[-2]), dtype=a.dtype)
        V = mnp.zeros((a.shape[-1], a.shape[-1]), dtype=a.dtype)
        return U, S, V
    else:
        # For PyTorch compatibility, we need to transpose V
        U, S, V = mnp.linalg.svd(a, full_matrices=not some)
        return U, S, mnp.transpose(V)


@register_function(torch.cdist)
def _cdist(x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    # Use MindSpore's cdist implementation
    return mnp.linalg.cdist(x1, x2, p=p)


@register_function(torch.lu)
def lu(A, **kwargs):
    # In MindSpore, we can use SVD as an approximation for LU decomposition
    # This is a simplified implementation
    U, S, V = mnp.linalg.svd(A)
    # Create a pseudo LU decomposition
    lu = U @ mnp.diag(S) @ V
    # Create pivot indices (identity permutation for simplicity)
    pivots = mnp.arange(min(A.shape), dtype=ms.int32)
    # Return a flag indicating success
    info = mnp.array(0, dtype=ms.int32)
    return lu, pivots, info


@register_function(torch.lu_solve)
def lu_solve(b, LU_data, LU_pivots, **kwargs):
    # In MindSpore, we can use solve as an approximation for lu_solve
    # This is a simplified implementation
    x = mnp.linalg.solve(LU_data, b)
    return x


@register_function(torch.linalg.tensorsolve)
def linalg_tensorsolve(A, b, dims=None):
    # In MindSpore, we can implement tensorsolve using reshaping and solve
    # This is a simplified implementation
    # If dims is provided, move those axes to the end
    if dims is not None:
        # Create a list of axes to move
        move_axes = list(dims)
        # Move them to the end
        for i, ax in enumerate(move_axes):
            A = mnp.moveaxis(A, ax, -(len(move_axes) - i))
        dims = None
    
    # Reshape b to match the leading dimensions of A
    if A.shape[: b.ndim] != b.shape:
        b = mnp.reshape(b, A.shape[: b.ndim])
    
    # For simplicity, we'll use a direct approach for small tensors
    # This is a placeholder implementation that may need refinement
    # depending on specific tensor shapes and use cases
    return mnp.linalg.solve(A.reshape(-1, A.shape[-1]), b.reshape(-1)).reshape(A.shape[1:])


@register_function(torch.nn.functional.linear)
def functional_linear(self, weights, bias=None):
    res = ms.ops.einsum("...a,ba->...b", self, weights)
    if bias is not None:
        res += bias
    return res


@register_function(torch.nn.functional.interpolate, is_mindspore_function=False)
def functional_interpolate(
    input,
    size: Tuple[int, int],
    scale_factor: Optional[float],
    mode: str,
    align_corners: bool,
    recompute_scale_factor: bool,
    antialias: bool,
):
    # MindSpore supported interpolation methods
    supported_methods = (
        "nearest",
        "linear",
        "bilinear",
        "trilinear",
    )
    
    # Map PyTorch mode names to MindSpore mode names
    mode_map = {
        "nearest": "nearest",
        "linear": "linear",
        "bilinear": "bilinear",
        "trilinear": "trilinear",
    }
    
    if mode not in mode_map:
        raise torch4ms.tensor.OperatorNotFound(
            f"MindSpore does not support interpolation mode: {mode}. Supported modes are: {list(mode_map.keys())}"
        )
    
    # None check
    antialias = antialias or False
    align_corners = align_corners or False
    
    # Use MindSpore's resize function
    # Note: This is a simplified implementation
    # For 4D input (N, C, H, W)
    if input.ndim == 4:
        # Transpose to (N, H, W, C) for MindSpore's image processing
        input_nhwc = mnp.transpose(input, (0, 2, 3, 1))
        # Resize using MindSpore's image.resize
        from mindspore import image
        resized = image.resize(input_nhwc, size, interpolation=mode_map[mode])
        # Transpose back to (N, C, H, W)
        return mnp.transpose(resized, (0, 3, 1, 2))
    else:
        # For other dimensions, use a more general approach
        raise torch4ms.tensor.OperatorNotFound(
            f"Interpolation for input dimension {input.ndim} not implemented yet"
        )


@register_function(torch.Tensor.repeat_interleave, is_mindspore_function=False)
def torch_Tensor_repeat_interleave(
    self, repeats, dim=None, *, output_size=None
):
    # Use MindSpore's repeat function
    # Note: MindSpore's repeat doesn't have total_repeat_length parameter
    # If output_size is provided, we'll use it to adjust the output
    result = mnp.repeat(self, repeats, axis=dim)
    
    # Handle output_size if provided
    if output_size is not None and result.shape[dim] != output_size:
        # This is a simplified handling
        # In practice, you might need more complex logic depending on the use case
        slice_idx = [slice(None)] * result.ndim
        slice_idx[dim] = slice(0, output_size)
        result = result[tuple(slice_idx)]
    
    return result


@register_function(torch.nn.functional.max_pool2d, is_mindspore_function=False)
def _functional_max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    
    # Use MindSpore's MaxPool2D operation
    max_pool = ms.ops.MaxPool2D(
        kernel_size=kernel_size,
        stride=stride,
        pad_mode='pad' if padding > 0 else 'valid',
        data_format='NCHW'
    )
    
    # Handle padding
    if padding > 0:
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)  # (top, bottom, left, right)
        input = mops.pad(input, padding)
    
    # Apply max pooling
    output = max_pool(input)
    
    # If return_indices is True, we need to return indices as well
    # Note: MindSpore's MaxPool2D doesn't support returning indices directly
    # This is a limitation in the current implementation
    if return_indices:
        # For now, return zeros as indices (this is not correct and should be improved)
        indices = mnp.zeros_like(output)
        return output, indices
    
    return output


@register_function(torch.nn.functional.conv1d)
def functional_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    将 torch.nn.functional.conv1d 转换为 aten.convolution
    参考 torchax 的实现方式，直接调用已注册的 _aten_convolution 函数来避免类型检查错误
    """
    # 确保参数格式正确
    if isinstance(stride, int):
        stride = (stride,)
    elif not isinstance(stride, (tuple, list)):
        stride = (stride,)
    else:
        stride = tuple(stride)
    
    if isinstance(padding, int):
        padding = (padding,)
    elif not isinstance(padding, (tuple, list)):
        padding = (padding,)
    else:
        padding = tuple(padding)
    
    if isinstance(dilation, int):
        dilation = (dilation,)
    elif not isinstance(dilation, (tuple, list)):
        dilation = (dilation,)
    else:
        dilation = tuple(dilation)
    
    # 参考 torchax 的实现，直接调用已注册的 _aten_convolution 函数
    # 这样可以避免 PyTorch 的类型检查器检查参数类型
    # 注意：aten.convolution 的参数顺序是: input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    result = _aten_convolution(input, weight, bias, stride, padding, dilation, False, (), groups)
    
    # 将结果转换回 torch4ms.Tensor（如果输入是 torch4ms.Tensor）
    if isinstance(input, torch4ms.tensor.Tensor):
        return input._env.ms2t_iso(result)
    return result


@register_function(torch.nn.functional.conv2d)
def functional_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    将 torch.nn.functional.conv2d 转换为 aten.convolution
    参考 torchax 的实现方式，直接调用已注册的 _aten_convolution 函数来避免类型检查错误
    """
    # 确保参数格式正确
    if isinstance(stride, int):
        stride = (stride, stride)
    elif not isinstance(stride, (tuple, list)):
        stride = (stride, stride)
    else:
        stride = tuple(stride)
    
    if isinstance(padding, int):
        padding = (padding, padding)
    elif not isinstance(padding, (tuple, list)):
        padding = (padding, padding)
    else:
        padding = tuple(padding)
    
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    elif not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation)
    else:
        dilation = tuple(dilation)
    
    # 参考 torchax 的实现，直接调用已注册的 _aten_convolution 函数
    # 这样可以避免 PyTorch 的类型检查器检查参数类型
    # 注意：aten.convolution 的参数顺序是: input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    result = _aten_convolution(input, weight, bias, stride, padding, dilation, False, (), groups)
    
    # 将结果转换回 torch4ms.Tensor（如果输入是 torch4ms.Tensor）
    if isinstance(input, torch4ms.tensor.Tensor):
        return input._env.ms2t_iso(result)
    return result


@register_function(torch.nn.functional.conv3d)
def functional_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    将 torch.nn.functional.conv3d 转换为 aten.convolution
    参考 torchax 的实现方式，直接调用已注册的 _aten_convolution 函数来避免类型检查错误
    """
    # 确保参数格式正确
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    elif not isinstance(stride, (tuple, list)):
        stride = (stride, stride, stride)
    else:
        stride = tuple(stride)
    
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    elif not isinstance(padding, (tuple, list)):
        padding = (padding, padding, padding)
    else:
        padding = tuple(padding)
    
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)
    elif not isinstance(dilation, (tuple, list)):
        dilation = (dilation, dilation, dilation)
    else:
        dilation = tuple(dilation)
    
    # 参考 torchax 的实现，直接调用已注册的 _aten_convolution 函数
    # 这样可以避免 PyTorch 的类型检查器检查参数类型
    # 注意：aten.convolution 的参数顺序是: input, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    result = _aten_convolution(input, weight, bias, stride, padding, dilation, False, (), groups)
    
    # 将结果转换回 torch4ms.Tensor（如果输入是 torch4ms.Tensor）
    if isinstance(input, torch4ms.tensor.Tensor):
        return input._env.ms2t_iso(result)
    return result
