from ..._tensor import Tensor


def _contiguous_stride(shape):
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.append(acc)
        acc *= d
    return tuple(reversed(stride))


def reshape(a, shape):
    shape = tuple(shape)
    size = 1
    for d in a.shape:
        size *= d
    new_size = 1
    for d in shape:
        new_size *= d
    if size != new_size:
        raise ValueError("reshape size mismatch")
    stride = _contiguous_stride(shape)
    return Tensor(a.storage(), shape, stride, a.offset)


def view(a, shape):
    return reshape(a, shape)


def transpose(a, dim0, dim1):
    shape = list(a.shape)
    stride = list(a.stride)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    return Tensor(a.storage(), shape, stride, a.offset)
