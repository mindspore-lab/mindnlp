"""PyBoost CPU backend using MindSpore primitives.

All ops are instantiated once at module load with CPU device set.
These are much faster than NumPy as they use optimized C++ kernels.
"""

from mindspore.ops.auto_generate.gen_ops_prim import (
    Add, Sub, Mul, Div, Neg, Abs,
    Pow, Exp, Log, Sqrt, Rsqrt,
    Sin, Cos, Tanh, Sigmoid,
    ReLU, GeLU, SiLU,
    MatMulExt, BatchMatMulExt,
    SumExt, MeanExt, MaxDim, MinDim, Max, Min,
    ProdExt,
    ArgMaxExt, ArgMinExt,
    Var, Std,
    Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual,
    Reshape, Transpose,
    Clone, Contiguous,
    OnesLikeExt, ZerosLikeExt, FullLike,
    Concat, StackExt,
    SplitWithSize, SplitTensor, Chunk,
    SelectV2,
    ReduceAll, Any,
    LogicalNot,
    ClampScalar, ClampTensor,
    Embedding, DropoutExt,
    TopkExt, MultinomialExt,
    # In-place ops
    InplaceAddExt, InplaceSubExt, InplaceMul, InplaceDiv,
    InplaceCopy, InplaceFillScalar, InplaceZero,
)

# Instantiate ops with CPU device
add_op = Add().set_device('CPU')
sub_op = Sub().set_device('CPU')
mul_op = Mul().set_device('CPU')
div_op = Div().set_device('CPU')
neg_op = Neg().set_device('CPU')
abs_op = Abs().set_device('CPU')

pow_op = Pow().set_device('CPU')
exp_op = Exp().set_device('CPU')
log_op = Log().set_device('CPU')
sqrt_op = Sqrt().set_device('CPU')
rsqrt_op = Rsqrt().set_device('CPU')

sin_op = Sin().set_device('CPU')
cos_op = Cos().set_device('CPU')
tanh_op = Tanh().set_device('CPU')
sigmoid_op = Sigmoid().set_device('CPU')

relu_op = ReLU().set_device('CPU')
gelu_op = GeLU().set_device('CPU')
silu_op = SiLU().set_device('CPU')

matmul_op = MatMulExt().set_device('CPU')
bmm_op = BatchMatMulExt().set_device('CPU')

sum_op = SumExt().set_device('CPU')
mean_op = MeanExt().set_device('CPU')
max_op = MaxDim().set_device('CPU')
min_op = MinDim().set_device('CPU')

equal_op = Equal().set_device('CPU')
not_equal_op = NotEqual().set_device('CPU')
greater_op = Greater().set_device('CPU')
less_op = Less().set_device('CPU')
greater_equal_op = GreaterEqual().set_device('CPU')
less_equal_op = LessEqual().set_device('CPU')

reshape_op = Reshape().set_device('CPU')
transpose_op = Transpose().set_device('CPU')

clone_op = Clone().set_device('CPU')
contiguous_op = Contiguous().set_device('CPU')

# Additional ops
ones_like_op = OnesLikeExt().set_device('CPU')
zeros_like_op = ZerosLikeExt().set_device('CPU')
full_like_op = FullLike().set_device('CPU')

concat_op = Concat().set_device('CPU')
stack_op = StackExt().set_device('CPU')
split_with_size_op = SplitWithSize().set_device('CPU')
split_tensor_op = SplitTensor().set_device('CPU')
chunk_op = Chunk().set_device('CPU')

select_op = SelectV2().set_device('CPU')
reduce_all_op = ReduceAll().set_device('CPU')
reduce_any_op = Any().set_device('CPU')
logical_not_op = LogicalNot().set_device('CPU')

clamp_scalar_op = ClampScalar().set_device('CPU')
clamp_tensor_op = ClampTensor().set_device('CPU')

embedding_op = Embedding().set_device('CPU')
dropout_ext_op = DropoutExt().set_device('CPU')

topk_ext_op = TopkExt().set_device('CPU')
multinomial_ext_op = MultinomialExt().set_device('CPU')

max_global_op = Max().set_device('CPU')
min_global_op = Min().set_device('CPU')
prod_op = ProdExt().set_device('CPU')
argmax_ext_op = ArgMaxExt().set_device('CPU')
argmin_ext_op = ArgMinExt().set_device('CPU')
var_op = Var().set_device('CPU')
std_op = Std().set_device('CPU')

# In-place ops
inplace_add_op = InplaceAddExt().set_device('CPU')
inplace_sub_op = InplaceSubExt().set_device('CPU')
inplace_mul_op = InplaceMul().set_device('CPU')
inplace_div_op = InplaceDiv().set_device('CPU')
inplace_copy_op = InplaceCopy().set_device('CPU')
inplace_fill_op = InplaceFillScalar().set_device('CPU')
inplace_zero_op = InplaceZero().set_device('CPU')


def _get_ms_data(tensor):
    """Extract MindSpore tensor from our Tensor, handling views correctly.

    Args:
        tensor: mindtorch_v2.Tensor or scalar

    Returns:
        mindspore.Tensor ready for pyboost ops
    """
    from .._tensor import Tensor
    import mindspore
    import numpy as np

    if isinstance(tensor, Tensor):
        ms_tensor = tensor._storage.ms_tensor

        # Check if tensor is contiguous
        if not tensor.is_contiguous():
            # Non-contiguous view (e.g., transpose) - need to create contiguous copy
            # Use numpy to handle the strided access correctly
            # Get the data as numpy, reshape using tensor's view
            storage_np = ms_tensor.asnumpy()
            # Apply stride-based indexing
            shape = tensor._shape
            stride = tensor._stride
            offset = tensor._storage_offset

            # Create strided view and copy to contiguous
            result_np = np.lib.stride_tricks.as_strided(
                storage_np[offset:],
                shape=shape,
                strides=tuple(s * storage_np.itemsize for s in stride)
            ).copy()
            return mindspore.Tensor(result_np)

        # Contiguous tensor - handle views with offset
        numel = 1
        for s in tensor._shape:
            numel *= s
        offset = tensor._storage_offset

        # If it's a view (has offset or different size), slice the storage
        if offset > 0 or ms_tensor.shape[0] != numel:
            storage_np = ms_tensor.asnumpy()
            # Check if slice would be valid
            if offset + numel > len(storage_np):
                # Storage too small - likely an error in tensor construction
                # Fall back to using strided access
                result_np = np.lib.stride_tricks.as_strided(
                    storage_np[offset:] if offset < len(storage_np) else storage_np,
                    shape=tensor._shape,
                    strides=tuple(s * storage_np.itemsize for s in tensor._stride)
                ).copy()
                return mindspore.Tensor(result_np)
            result_np = storage_np[offset:offset + numel].reshape(tensor._shape)
            return mindspore.Tensor(result_np)

        # Simple case - reshape if needed
        if ms_tensor.shape != tensor.shape:
            ms_tensor = ms_tensor.reshape(tensor.shape)
        return ms_tensor
    elif isinstance(tensor, (int, float, bool)):
        return mindspore.Tensor(tensor)
    elif isinstance(tensor, np.ndarray):
        return mindspore.Tensor(tensor)
    elif isinstance(tensor, (np.floating, np.integer, np.bool_)):
        # Handle numpy scalar types - convert to Python native type first
        return mindspore.Tensor(tensor.item())
    elif isinstance(tensor, mindspore.Tensor):
        return tensor
    else:
        raise TypeError(f"Cannot convert {type(tensor)} to MindSpore tensor")


def _wrap_result(ms_tensor, device="cpu"):
    """Wrap MindSpore tensor result in our Tensor.

    Args:
        ms_tensor: mindspore.Tensor result from pyboost op
        device: device string

    Returns:
        mindtorch_v2.Tensor
    """
    from .._tensor import Tensor
    from .._storage import TypedStorage
    from .. import _dtype as dtype_mod
    from .._device import device as device_cls

    # Create storage from flattened tensor
    flat = ms_tensor.reshape(-1)
    storage = TypedStorage.__new__(TypedStorage)
    storage._ms_tensor = flat
    storage._size = flat.shape[0]
    storage._dtype = dtype_mod.from_mindspore_dtype(flat.dtype)
    storage._device = device_cls(device)

    # Create tensor with proper shape
    return Tensor(
        _storage=storage,
        _shape=tuple(ms_tensor.shape),
        _stride=None,
        _storage_offset=0
    )
