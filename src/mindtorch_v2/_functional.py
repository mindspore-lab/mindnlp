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


def sign(a):
    return dispatch("sign", a.device.type, a)


def signbit(a):
    return dispatch("signbit", a.device.type, a)


def isnan(a):
    return dispatch("isnan", a.device.type, a)


def isinf(a):
    return dispatch("isinf", a.device.type, a)


def isfinite(a):
    return dispatch("isfinite", a.device.type, a)

def sinh(a):
    return dispatch("sinh", a.device.type, a)


def cosh(a):
    return dispatch("cosh", a.device.type, a)


def asinh(a):
    return dispatch("asinh", a.device.type, a)


def acosh(a):
    return dispatch("acosh", a.device.type, a)


def atanh(a):
    return dispatch("atanh", a.device.type, a)


def erf(a):
    return dispatch("erf", a.device.type, a)


def erfc(a):
    return dispatch("erfc", a.device.type, a)


def softplus(a):
    return dispatch("softplus", a.device.type, a)


def clamp(a, min_val=None, max_val=None):
    return dispatch("clamp", a.device.type, a, min_val, max_val)


def clamp_min(a, min_val):
    return dispatch("clamp_min", a.device.type, a, min_val)


def clamp_max(a, max_val):
    return dispatch("clamp_max", a.device.type, a, max_val)


def relu6(a):
    return dispatch("relu6", a.device.type, a)


def hardtanh(a, min_val=-1.0, max_val=1.0):
    return dispatch("hardtanh", a.device.type, a, min_val, max_val)


def min(a, b):
    return dispatch("min", a.device.type, a, b)


def max(a, b):
    return dispatch("max", a.device.type, a, b)


def amin(a, dim=None, keepdim=False):
    return dispatch("amin", a.device.type, a, dim=dim, keepdim=keepdim)


def amax(a, dim=None, keepdim=False):
    return dispatch("amax", a.device.type, a, dim=dim, keepdim=keepdim)


def fmin(a, b):
    return dispatch("fmin", a.device.type, a, b)


def fmax(a, b):
    return dispatch("fmax", a.device.type, a, b)


def where(cond, x, y):
    return dispatch("where", x.device.type, cond, x, y)


def atan(a):
    return dispatch("atan", a.device.type, a)


def atan2(a, b):
    return dispatch("atan2", a.device.type, a, b)


def asin(a):
    return dispatch("asin", a.device.type, a)


def acos(a):
    return dispatch("acos", a.device.type, a)


def lerp(a, b, weight):
    return dispatch("lerp", a.device.type, a, b, weight)


def addcmul(a, b, c, value=1.0):
    return dispatch("addcmul", a.device.type, a, b, c, value=value)


def addcdiv(a, b, c, value=1.0):
    return dispatch("addcdiv", a.device.type, a, b, c, value=value)


def logaddexp(a, b):
    return dispatch("logaddexp", a.device.type, a, b)


def logaddexp2(a, b):
    return dispatch("logaddexp2", a.device.type, a, b)


def hypot(a, b):
    return dispatch("hypot", a.device.type, a, b)


def remainder(a, b):
    return dispatch("remainder", a.device.type, a, b)


def fmod(a, b):
    return dispatch("fmod", a.device.type, a, b)

def sinh(a):
    return dispatch("sinh", a.device.type, a)


def cosh(a):
    return dispatch("cosh", a.device.type, a)


def erf(a):
    return dispatch("erf", a.device.type, a)


def erfc(a):
    return dispatch("erfc", a.device.type, a)


def softplus(a):
    return dispatch("softplus", a.device.type, a)


def sum(a, dim=None, keepdim=False):
    return dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)


def all(a, dim=None, keepdim=False):
    return dispatch("all", a.device.type, a, dim=dim, keepdim=keepdim)


def any(a, dim=None, keepdim=False):
    return dispatch("any", a.device.type, a, dim=dim, keepdim=keepdim)


def argmax(a, dim=None, keepdim=False):
    return dispatch("argmax", a.device.type, a, dim=dim, keepdim=keepdim)


def argmin(a, dim=None, keepdim=False):
    return dispatch("argmin", a.device.type, a, dim=dim, keepdim=keepdim)


def count_nonzero(a, dim=None, keepdim=False):
    return dispatch("count_nonzero", a.device.type, a, dim=dim, keepdim=keepdim)


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return dispatch("allclose", a.device.type, a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return dispatch("isclose", a.device.type, a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def equal(a, b):
    return dispatch("equal", a.device.type, a, b)


def logspace(start, end, steps, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("logspace", dev, start, end, steps, dtype=dtype)


def eye(n, m=None, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("eye", dev, n, m, dtype=dtype)


def range(start, end, step=1, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("range", dev, start, end, step, dtype=dtype)


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


def arange(start, end=None, step=1, dtype=None, device=None):
    dev = _as_device(device)
    if end is None:
        start, end = 0, start
    return dispatch("arange", dev, start, end, step, dtype=dtype)


def linspace(start, end, steps, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("linspace", dev, start, end, steps, dtype=dtype)


def full(shape, fill_value, dtype=None, device=None):
    dev = _as_device(device)
    return dispatch("full", dev, shape, fill_value, dtype=dtype)


def to(a, device, non_blocking=False):
    return dispatch("to", a.device, a, device, non_blocking=non_blocking)



def _as_device(dev):
    if dev is None:
        return get_default_device()
    if isinstance(dev, str):
        return Device(dev)
    return dev
