from mindspore import ops

def logit(input, eps=None, *, out=None):
    return ops.logit(input, eps)
