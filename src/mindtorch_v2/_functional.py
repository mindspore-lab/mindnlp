from ._dispatch.dispatcher import dispatch


def add(a, b):
    return dispatch("add", a.device.type, a, b)


def mul(a, b):
    return dispatch("mul", a.device.type, a, b)


def matmul(a, b):
    return dispatch("matmul", a.device.type, a, b)


def relu(a):
    return dispatch("relu", a.device.type, a)


def sum(a, dim=None, keepdim=False):
    return dispatch("sum", a.device.type, a, dim=dim, keepdim=keepdim)
