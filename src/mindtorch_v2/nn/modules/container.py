from ..module import Module
from ..parameter import Parameter


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():
                self._modules[key] = module
        else:
            for idx, module in enumerate(args):
                self._modules[str(idx)] = module

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx):
        keys = list(self._modules.keys())
        if isinstance(idx, slice):
            return Sequential(*[self._modules[k] for k in keys[idx]])
        return self._modules[keys[idx]]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def append(self, module):
        self._modules[str(len(self._modules))] = module
        return self


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self._modules[str(idx)] = module

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(list(self._modules.values())[idx])
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __setitem__(self, idx, module):
        if idx < 0:
            idx += len(self)
        self._modules[str(idx)] = module

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __contains__(self, module):
        return module in self._modules.values()

    def append(self, module):
        self._modules[str(len(self._modules))] = module
        return self

    def extend(self, modules):
        for module in modules:
            self.append(module)
        return self

    def insert(self, index, module):
        items = list(self._modules.values())
        items.insert(index, module)
        self._modules.clear()
        for i, m in enumerate(items):
            self._modules[str(i)] = m

    def forward(self):
        raise NotImplementedError("ModuleList is not callable")


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            if isinstance(modules, dict):
                for key, module in modules.items():
                    self._modules[key] = module
            else:
                for key, module in modules:
                    self._modules[key] = module

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self._modules[key] = module

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules):
        if isinstance(modules, dict):
            for key, module in modules.items():
                self._modules[key] = module
        else:
            for key, module in modules:
                self._modules[key] = module
        return self

    def forward(self):
        raise NotImplementedError("ModuleDict is not callable")


class ParameterList(Module):
    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            for idx, p in enumerate(parameters):
                self.register_parameter(str(idx), p)

    def __getitem__(self, idx):
        keys = list(self._parameters.keys())
        if idx < 0:
            idx += len(self)
        return self._parameters[keys[idx]]

    def __setitem__(self, idx, param):
        keys = list(self._parameters.keys())
        if idx < 0:
            idx += len(self)
        self.register_parameter(keys[idx], param)

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters.values())

    def append(self, param):
        self.register_parameter(str(len(self._parameters)), param)
        return self

    def forward(self):
        raise NotImplementedError("ParameterList is not callable")


class ParameterDict(Module):
    def __init__(self, parameters=None):
        super().__init__()
        if parameters is not None:
            if isinstance(parameters, dict):
                for key, param in parameters.items():
                    self.register_parameter(key, param)
            else:
                for key, param in parameters:
                    self.register_parameter(key, param)

    def __getitem__(self, key):
        return self._parameters[key]

    def __setitem__(self, key, param):
        self.register_parameter(key, param)

    def __delitem__(self, key):
        del self._parameters[key]

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def __contains__(self, key):
        return key in self._parameters

    def keys(self):
        return self._parameters.keys()

    def values(self):
        return self._parameters.values()

    def items(self):
        return self._parameters.items()

    def forward(self):
        raise NotImplementedError("ParameterDict is not callable")
