"""Module base class for neural network layers."""

from typing import Iterator, Tuple, Dict, Optional, Any, Set
from collections import OrderedDict

from .parameter import Parameter
from .._tensor import Tensor


class Module:
    """Base class for all neural network modules.

    Subclasses should implement the forward() method.
    """

    _version: int = 1
    training: bool

    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' function"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        # Handle Parameter assignment
        if isinstance(value, Parameter):
            if '_parameters' not in self.__dict__:
                self._parameters = OrderedDict()
            self._parameters[name] = value
        elif isinstance(value, Module):
            if '_modules' not in self.__dict__:
                self._modules = OrderedDict()
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        """Add a buffer to the module."""
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Add a parameter to the module."""
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def add_module(self, name: str, module: Optional['Module']) -> None:
        """Add a child module."""
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters with names."""
        memo: Set[int] = set()
        for name, p in self._parameters.items():
            if p is not None and id(p) not in memo:
                memo.add(id(p))
                yield prefix + name, p
        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    for name, p in module.named_parameters(prefix=submodule_prefix, recurse=True):
                        if id(p) not in memo:
                            memo.add(id(p))
                            yield name, p

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Return an iterator over module buffers."""
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Return an iterator over module buffers with names."""
        memo: Set[int] = set()
        for name, b in self._buffers.items():
            if b is not None and id(b) not in memo:
                memo.add(id(b))
                yield prefix + name, b
        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    for name, b in module.named_buffers(prefix=submodule_prefix, recurse=True):
                        if id(b) not in memo:
                            memo.add(id(b))
                            yield name, b

    def children(self) -> Iterator['Module']:
        """Return an iterator over immediate child modules."""
        for name, module in self._modules.items():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over immediate child modules with names."""
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def modules(self) -> Iterator['Module']:
        """Return an iterator over all modules in the network."""
        for name, module in self.named_modules():
            yield module

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over all modules with names."""
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(prefix=submodule_prefix):
                    yield m

    def train(self, mode: bool = True) -> 'Module':
        """Set the module in training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """Set the module in evaluation mode."""
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True) -> 'Module':
        """Set requires_grad for all parameters."""
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_grad_()

    def to(self, device=None, dtype=None):
        """Move module to device/dtype (placeholder - returns self)."""
        # TODO: Implement actual device/dtype conversion
        return self

    def cuda(self, device=None):
        """Move module to CUDA (no-op in mindtorch_v2, returns self)."""
        return self

    def cpu(self):
        """Move module to CPU (no-op in mindtorch_v2, returns self)."""
        return self

    def xpu(self, device=None):
        """Move module to XPU (no-op in mindtorch_v2, returns self)."""
        return self

    def type(self, dst_type):
        """Cast module parameters to dst_type (placeholder - returns self)."""
        return self

    def float(self):
        """Cast module to float32 (placeholder - returns self)."""
        return self

    def double(self):
        """Cast module to float64 (placeholder - returns self)."""
        return self

    def half(self):
        """Cast module to float16 (placeholder - returns self)."""
        return self

    def bfloat16(self):
        """Cast module to bfloat16 (placeholder - returns self)."""
        return self

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + '(']
        for name, module in self._modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f'  ({name}): {mod_str}')
        lines.append(')')
        return '\n'.join(lines) if len(self._modules) > 0 else f'{self.__class__.__name__}()'

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return ''
