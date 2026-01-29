"""PyBoost op registry.

All pyboost primitives are instantiated once with device set.
Use get_pyboost_op(name) to get the op instance.

NEVER use mindspore.ops or mindspore.mint directly.
"""
from typing import Optional, Dict, Any
from functools import lru_cache

from mindspore.ops.auto_generate.gen_ops_prim import (
    # Binary math
    Add, Sub, Mul, Div, Pow,
    # Unary math
    Neg, Abs, Exp, Log, Sqrt, Rsqrt,
    Sin, Cos, Tanh, Sigmoid,
    # Activations
    ReLU, GeLU, SiLU,
    # Matrix ops
    MatMulExt, BatchMatMulExt,
    # Reductions
    SumExt, MeanExt, MaxDim, MinDim,
    ProdExt, ArgMaxExt, ArgMinExt,
    # Comparison
    Equal, NotEqual, Greater, Less, GreaterEqual, LessEqual,
    # Shape ops
    Reshape, Transpose,
    # Other
    Concat, StackExt, ReduceAll,
)

# Device for all ops
_DEVICE = 'CPU'


@lru_cache(maxsize=128)
def _create_op(op_class, device: str = _DEVICE):
    """Create and cache a pyboost op instance."""
    return op_class().set_device(device)


# Op name to class mapping
_OP_MAP: Dict[str, Any] = {
    # Binary math
    'add': Add,
    'sub': Sub,
    'mul': Mul,
    'div': Div,
    'pow': Pow,
    # Unary math
    'neg': Neg,
    'abs': Abs,
    'exp': Exp,
    'log': Log,
    'sqrt': Sqrt,
    'rsqrt': Rsqrt,
    'sin': Sin,
    'cos': Cos,
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    # Activations
    'relu': ReLU,
    'gelu': GeLU,
    'silu': SiLU,
    # Matrix ops
    'matmul': MatMulExt,
    'bmm': BatchMatMulExt,
    # Reductions
    'sum': SumExt,
    'mean': MeanExt,
    'max_dim': MaxDim,
    'min_dim': MinDim,
    'prod': ProdExt,
    'argmax': ArgMaxExt,
    'argmin': ArgMinExt,
    # Comparison
    'eq': Equal,
    'ne': NotEqual,
    'gt': Greater,
    'lt': Less,
    'ge': GreaterEqual,
    'le': LessEqual,
    # Shape ops
    'reshape': Reshape,
    'transpose': Transpose,
    # Other
    'concat': Concat,
    'stack': StackExt,
    'reduce_all': ReduceAll,
}


def get_pyboost_op(name: str, device: str = _DEVICE) -> Optional[Any]:
    """Get a pyboost op by name.

    Args:
        name: Op name (e.g., 'add', 'matmul')
        device: Device to set ('CPU', 'GPU', 'Ascend')

    Returns:
        Pyboost op instance or None if not found
    """
    op_class = _OP_MAP.get(name)
    if op_class is None:
        return None
    return _create_op(op_class, device)


def set_device(device: str):
    """Set default device for all ops."""
    global _DEVICE
    _DEVICE = device
    _create_op.cache_clear()
