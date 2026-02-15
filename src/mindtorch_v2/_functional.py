from ._dispatch.dispatcher import dispatch
from ._autograd.grad_mode import GradMode, no_grad
from ._device import device as Device, get_default_device
from ._dtype import to_numpy_dtype


def add(a, b):
    return dispatch("add", a.device.type, a, b)


def mul(a, b):
    return dispatch("mul", a.device.type, a, b)


def matmul(a, b):
    return dispatch("matmul", a.device.type, a, b)


def relu(a):
    return dispatch("relu", a.device.type, a)


def abs(a):
    return dispatch("abs", a.device.type, a)


def neg(a):
    return dispatch("neg", a.device.type, a)


def exp(a):
    return dispatch("exp", a.device.type, a)


def log(a):
    return dispatch("log", a.device.type, a)


def sqrt(a):
    return dispatch("sqrt", a.device.type, a)


def sin(a):
    return dispatch("sin", a.device.type, a)


def cos(a):
    return dispatch("cos", a.device.type, a)


def tan(a):
    return dispatch("tan", a.device.type, a)


def tanh(a):
    return dispatch("tanh", a.device.type, a)


def sigmoid(a):
    return dispatch("sigmoid", a.device.type, a)


def floor(a):
    return dispatch("floor", a.device.type, a)


def ceil(a):
    return dispatch("ceil", a.device.type, a)


def round(a):
    return dispatch("round", a.device.type, a)


def trunc(a):
    return dispatch("trunc", a.device.type, a)


def frac(a):
    return dispatch("frac", a.device.type, a)


def pow(a, b):
    return dispatch("pow", a.device.type, a, b)


def log2(a):
    return dispatch("log2", a.device.type, a)


def log10(a):
    return dispatch("log10", a.device.type, a)


def exp2(a):
    return dispatch("exp2", a.device.type, a)


def rsqrt(a):
    return dispatch("rsqrt", a.device.type, a)


def sum(a, dim=None, keepdim=False):
    return dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)


def reshape(a, shape):
    return dispatch("reshape", a.device.type, a, shape)


def view(a, shape):
    return reshape(a, shape)


def transpose(a, dim0, dim1):
    return dispatch("transpose", a.device.type, a, dim0, dim1)


def tensor(data, dtype=None, device=None, requires_grad=False):
    dev = _as_device(device)
    return dispatch("tensor", dev, data, dtype=dtype, requires_grad=requires_grad)


def zeros(shape, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("zeros", dev, shape, dtype=dtype)


def ones(shape, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("ones", dev, shape, dtype=dtype)


def empty(shape, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("empty", dev, shape, dtype=dtype)


def to(a, device, non_blocking=False):
    return dispatch("to", a.device, a, device, non_blocking=non_blocking)



def _as_device(dev):
    if dev is None:
        return get_default_device()
    if isinstance(dev, str):
        return Device(dev)
    return dev
