from .parameter import Parameter


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
