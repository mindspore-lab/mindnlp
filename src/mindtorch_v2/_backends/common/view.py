from ..._tensor import Tensor


def _get_base(tensor):
    return tensor._base if tensor._base is not None else tensor


def _make_view(base, shape, stride, offset, op):
    view = Tensor(base.storage(), shape, stride, offset, requires_grad=base.requires_grad)
    view._base = base
    view._version_counter = base._version_counter
    view._view_meta = {
        "op": op,
        "shape": tuple(shape),
        "stride": tuple(stride),
        "offset": int(offset),
    }
    return view


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
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "reshape")


def view(a, shape):
    if isinstance(shape, int):
        shape = (shape,)
    else:
        shape = tuple(shape)
    size = 1
    for d in a.shape:
        size *= d
    new_size = 1
    for d in shape:
        new_size *= d
    if size != new_size:
        raise ValueError("view size mismatch")
    stride = _contiguous_stride(shape)
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "view")


def transpose(a, dim0, dim1):
    shape = list(a.shape)
    stride = list(a.stride)
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    stride[dim0], stride[dim1] = stride[dim1], stride[dim0]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "transpose")


def squeeze(a, dim=None):
    shape = list(a.shape)
    stride = list(a.stride)
    if dim is not None:
        d = dim if dim >= 0 else dim + len(shape)
        if 0 <= d < len(shape) and shape[d] == 1:
            del shape[d]
            del stride[d]
    else:
        pairs = [(s, st) for s, st in zip(shape, stride) if s != 1]
        shape = [p[0] for p in pairs]
        stride = [p[1] for p in pairs]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "squeeze")


def unsqueeze(a, dim):
    ndim = len(a.shape)
    d = dim if dim >= 0 else dim + ndim + 1
    shape = list(a.shape)
    stride = list(a.stride)
    new_stride = stride[d] * shape[d] if d < ndim else 1
    shape.insert(d, 1)
    stride.insert(d, new_stride)
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "unsqueeze")


def permute(a, dims):
    shape = [a.shape[d] for d in dims]
    stride = [a.stride[d] for d in dims]
    base = _get_base(a)
    return _make_view(base, shape, stride, a.offset, "permute")
