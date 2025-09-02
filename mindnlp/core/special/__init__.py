from ..executor import execute

def logit(input, eps=None, *, out=None):
    return execute('logit', input, eps)
