from .._tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        if isinstance(data, Tensor):
            storage = data.storage
            shape = data.shape
            stride = data.stride
            offset = data.offset
        else:
            raise TypeError("Parameter requires a Tensor")
        super().__init__(storage, shape, stride, offset, requires_grad=True)
