import numpy as np

from .._storage import typed_storage_from_numpy


def reduce_grad(grad, shape):
    """Reduce gradient to match the target shape by summing broadcast dimensions."""
    if grad.shape == shape:
        return grad

    if grad.device.type != "cpu":
        return _reduce_grad_dispatch(grad, shape)

    arr = grad.storage().data
    while arr.ndim > len(shape):
        arr = arr.sum(axis=0)
    for i, (g_dim, s_dim) in enumerate(zip(arr.shape, shape)):
        if s_dim == 1 and g_dim != 1:
            arr = arr.sum(axis=i, keepdims=True)
    storage = typed_storage_from_numpy(arr, grad.dtype)
    stride = tuple(np.array(arr.strides) // arr.itemsize)
    from .._tensor import Tensor
    return Tensor(storage, arr.shape, stride)


def _reduce_grad_dispatch(grad, shape):
    """reduce_grad using dispatched ops (works on NPU)."""
    from .._functional import sum as torch_sum
    from .grad_mode import no_grad

    result = grad
    with no_grad():
        while len(result.shape) > len(shape):
            result = torch_sum(result, dim=0)
        for i, (g_dim, s_dim) in enumerate(zip(result.shape, shape)):
            if s_dim == 1 and g_dim != 1:
                result = torch_sum(result, dim=i, keepdim=True)
    return result
