"""Container modules."""

from typing import Iterator, Iterable, Optional, overload
from collections import OrderedDict

from ..module import Module


class Sequential(Module):
    """A sequential container that chains modules."""

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None: ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleList(Module):
    """Holds submodules in a list."""

    def __init__(self, modules: Optional[Iterable[Module]] = None):
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def append(self, module: Module) -> 'ModuleList':
        """Append a module to the list."""
        self.add_module(str(len(self._modules)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> 'ModuleList':
        """Extend with modules from iterable."""
        for module in modules:
            self.append(module)
        return self

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        key = list(self._modules.keys())[idx]
        self._modules[key] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())


class ModuleDict(Module):
    """Holds submodules in a dict."""

    def __init__(self, modules: Optional[dict] = None):
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def update(self, modules: dict) -> None:
        for key, module in modules.items():
            self[key] = module
