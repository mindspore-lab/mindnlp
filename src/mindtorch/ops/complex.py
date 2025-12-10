from mindtorch.executor import execute

def real(input):
    return execute('real', input)

def imag(input):
    return execute('imag', input)

__all__ = ['real', 'imag']