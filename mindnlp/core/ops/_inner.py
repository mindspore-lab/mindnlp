"""inner ops"""
from mindnlp.core.executor import execute

def cast(input, dtype):
    return execute('cast', input, dtype)

def assign(input, other):
    return execute('assign', input, other)
__all__ = ['cast', 'assign']
