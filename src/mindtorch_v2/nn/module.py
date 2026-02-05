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
        self._non_persistent_buffers_set: Set[str] = set()
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
        elif isinstance(value, Tensor):
            # If setting a Tensor to a name that was previously a Parameter,
            # update _parameters as well (needed for weight tying)
            if '_parameters' in self.__dict__ and name in self._parameters:
                self._parameters[name] = value
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
        """Add a buffer to the module.

        Args:
            name: Name of the buffer
            tensor: Tensor to register (or None)
            persistent: If True, buffer will be part of state_dict. Default: True.
        """
        if '_buffers' not in self.__dict__:
            self._buffers = OrderedDict()
        if '_non_persistent_buffers_set' not in self.__dict__:
            self._non_persistent_buffers_set = set()

        self._buffers[name] = tensor

        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

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

    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters with names.

        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, recursively include parameters from submodules
            remove_duplicate: If True (default), remove duplicate parameters
        """
        memo: Set[int] = set() if remove_duplicate else None
        for name, p in self._parameters.items():
            if p is not None:
                if memo is None or id(p) not in memo:
                    if memo is not None:
                        memo.add(id(p))
                    yield prefix + name, p
        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    for name, p in module.named_parameters(prefix=submodule_prefix, recurse=True, remove_duplicate=remove_duplicate):
                        if memo is None or id(p) not in memo:
                            if memo is not None:
                                memo.add(id(p))
                            yield name, p

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Return an iterator over module buffers."""
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Tensor]]:
        """Return an iterator over module buffers with names."""
        memo: Set[int] = set() if remove_duplicate else None
        for name, b in self._buffers.items():
            if b is not None:
                if memo is None or id(b) not in memo:
                    if memo is not None:
                        memo.add(id(b))
                    yield prefix + name, b
        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    for name, b in module.named_buffers(prefix=submodule_prefix, recurse=True, remove_duplicate=remove_duplicate):
                        if memo is None or id(b) not in memo:
                            if memo is not None:
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

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True) -> Iterator[Tuple[str, 'Module']]:
        """Return an iterator over all modules with names."""
        if memo is None:
            memo = set()
        if remove_duplicate:
            if self in memo:
                return
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo=memo, prefix=submodule_prefix, remove_duplicate=remove_duplicate):
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

    def to(self, *args, **kwargs):
        """Move module to device and/or change dtype.

        Args:
            device: Target device ('cpu', 'cuda', 'npu', etc.)
            dtype: Target dtype

        Can be called as:
            module.to(device)
            module.to(dtype)
            module.to(device, dtype)
            module.to(device=device, dtype=dtype)
        """
        from .._device import device as device_cls
        from .. import dtype as dtype_mod

        # Parse args
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)

        for arg in args:
            if isinstance(arg, device_cls):
                device = arg
            elif isinstance(arg, str):
                if arg in ('cpu', 'cuda', 'npu', 'mps') or ':' in arg:
                    device = device_cls(arg)
                else:
                    # Might be dtype string
                    try:
                        dtype = getattr(dtype_mod, arg, None)
                    except:
                        pass
            elif isinstance(arg, dtype_mod.DType):
                dtype = arg

        # Move all parameters
        for name, param in self._parameters.items():
            if param is not None:
                new_param = param.to(device=device, dtype=dtype)
                # Preserve Parameter type
                if hasattr(param, 'requires_grad'):
                    new_param.requires_grad_(param.requires_grad)
                self._parameters[name] = new_param
                object.__setattr__(self, name, new_param)

        # Move all buffers
        for name, buf in self._buffers.items():
            if buf is not None:
                # Handle meta buffers specially - they need proper initialization
                if hasattr(buf, 'device') and hasattr(buf.device, 'type') and buf.device.type == 'meta':
                    if device is not None and (not hasattr(device, 'type') or device.type != 'meta'):
                        # Meta → real: reinitialize based on buffer name
                        import numpy as np
                        from .._dtype import dtype_to_numpy
                        np_dtype = dtype_to_numpy(buf.dtype)
                        if 'position_ids' in name:
                            # position_ids should be arange(0, seq_len)
                            total = 1
                            for d in buf.shape:
                                total *= d
                            arr = np.arange(total, dtype=np_dtype).reshape(buf.shape)
                        else:
                            arr = np.zeros(buf.shape, dtype=np_dtype)
                        from .._tensor import Tensor
                        new_buf = Tensor(arr, dtype=buf.dtype, device=str(device))
                        self._buffers[name] = new_buf
                        object.__setattr__(self, name, new_buf)
                        continue
                new_buf = buf.to(device=device, dtype=dtype)
                self._buffers[name] = new_buf
                object.__setattr__(self, name, new_buf)

        # Recursively apply to child modules
        for module in self._modules.values():
            if module is not None:
                module.to(*args, **kwargs)

        return self

    def cuda(self, device=None):
        """Move module to CUDA/NPU device."""
        if device is None:
            device = 'npu:0'  # Default to first NPU
        elif isinstance(device, int):
            device = f'npu:{device}'
        return self.to(device=device)

    def cpu(self):
        """Move module to CPU."""
        return self.to(device='cpu')

    def xpu(self, device=None):
        """Move module to XPU (maps to NPU in mindtorch_v2)."""
        return self.cuda(device)

    def type(self, dst_type):
        """Cast module parameters to dst_type."""
        from .. import dtype as dtype_mod
        if isinstance(dst_type, str):
            dtype_map = {
                'torch.FloatTensor': dtype_mod.float32,
                'torch.DoubleTensor': dtype_mod.float64,
                'torch.HalfTensor': dtype_mod.float16,
                'torch.BFloat16Tensor': dtype_mod.bfloat16,
                'torch.LongTensor': dtype_mod.int64,
                'torch.IntTensor': dtype_mod.int32,
                'torch.cuda.FloatTensor': dtype_mod.float32,
                'torch.cuda.DoubleTensor': dtype_mod.float64,
                'torch.cuda.HalfTensor': dtype_mod.float16,
            }
            dtype = dtype_map.get(dst_type)
            if dtype is not None:
                return self.to(dtype=dtype)
        return self

    def float(self):
        """Cast module to float32."""
        from .. import float32
        return self.to(dtype=float32)

    def double(self):
        """Cast module to float64."""
        from .. import float64
        return self.to(dtype=float64)

    def half(self):
        """Cast module to float16."""
        from .. import float16
        return self.to(dtype=float16)

    def bfloat16(self):
        """Cast module to bfloat16."""
        from .. import bfloat16
        return self.to(dtype=bfloat16)

    def to_empty(self, *, device=None, recurse=True):
        """Move module to device with empty parameters/buffers.

        This is used for meta device initialization - parameters are empty but have correct shape/dtype.
        """
        return self.to(device=device)

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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return a dictionary containing the whole state of the module.

        Both parameters and persistent buffers are included.
        Non-persistent buffers are excluded.
        """
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            if param is not None:
                # Check if __dict__ has a different value (e.g., from weight tying)
                # If so, use __dict__ value since that's what the module actually uses
                actual_param = self.__dict__.get(name, param)
                destination[prefix + name] = actual_param if keep_vars else actual_param.data

        # Only include persistent buffers
        non_persistent = getattr(self, '_non_persistent_buffers_set', set())
        for name, buf in self._buffers.items():
            if buf is not None and name not in non_persistent:
                destination[prefix + name] = buf if keep_vars else buf.data

        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)

        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load a state dictionary into the module.

        Args:
            state_dict: Dictionary containing state to load
            strict: If True, requires keys in state_dict to match module's keys
            assign: If True, assign instead of copy (preserves requires_grad)
        """
        missing_keys = []
        unexpected_keys = []
        size_mismatch_keys = []

        # Get all expected keys
        own_state = self.state_dict()
        expected_keys = set(own_state.keys())
        loaded_keys = set(state_dict.keys())

        if strict:
            missing_keys = list(expected_keys - loaded_keys)
            unexpected_keys = list(loaded_keys - expected_keys)

        # Track which modules had parameters loaded
        loaded_modules = set()

        # Load parameters and buffers
        for key in state_dict.keys():
            if key in expected_keys:
                # Find the module/parameter to update
                parts = key.split('.')
                module = self
                for part in parts[:-1]:
                    module = getattr(module, part)
                param_name = parts[-1]

                if param_name in module._parameters and module._parameters[param_name] is not None:
                    param = module._parameters[param_name]
                    new_value = state_dict[key]
                    # Check shape mismatch
                    if param.shape != new_value.shape:
                        size_mismatch_keys.append((key, param.shape, new_value.shape))
                        continue
                    if assign:
                        # Update both _parameters and __dict__ to handle tied weights
                        module._parameters[param_name] = new_value
                        if param_name in module.__dict__:
                            module.__dict__[param_name] = new_value
                    else:
                        param.data = new_value
                    # Mark the module as having loaded weights
                    loaded_modules.add(id(module))
                elif param_name in module._buffers and module._buffers[param_name] is not None:
                    buf = module._buffers[param_name]
                    new_value = state_dict[key]
                    # Check shape mismatch for buffers too
                    if buf.shape != new_value.shape:
                        size_mismatch_keys.append((key, buf.shape, new_value.shape))
                        continue
                    if assign:
                        module._buffers[param_name] = new_value
                    else:
                        module._buffers[param_name].data = new_value
                    loaded_modules.add(id(module))

        # Mark all modules that had parameters loaded as initialized
        # This prevents _init_weights from overwriting them
        def mark_loaded_modules(m):
            if id(m) in loaded_modules:
                m._skip_init = True
            for child in m.children():
                mark_loaded_modules(child)

        mark_loaded_modules(self)

        # Raise error for size mismatches first
        if size_mismatch_keys:
            error_msgs = []
            for key, expected_shape, loaded_shape in size_mismatch_keys:
                error_msgs.append(
                    f'size mismatch for {key}: copying a param with shape {loaded_shape} '
                    f'from checkpoint, the shape in current model is {expected_shape}.'
                )
            raise RuntimeError(
                'Error(s) in loading state_dict for {}:\n\t{}'.format(
                    self.__class__.__name__, '\n\t'.join(error_msgs)
                )
            )

        if strict and (missing_keys or unexpected_keys):
            error_msg = ''
            if missing_keys:
                error_msg += f'Missing key(s) in state_dict: {missing_keys}. '
            if unexpected_keys:
                error_msg += f'Unexpected key(s) in state_dict: {unexpected_keys}.'
            raise RuntimeError(error_msg)

        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _apply(self, fn):
        """Apply a function recursively to all parameters and buffers."""
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def get_submodule(self, target: str) -> 'Module':
        """Return the submodule given by target path.

        Args:
            target: Dot-separated path to the submodule (e.g. 'layer.0.attention')
        """
        if target == '':
            return self

        atoms = target.split('.')
        mod = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    f"Module '{type(mod).__name__}' has no attribute '{item}'"
                )
            mod = getattr(mod, item)
            if not isinstance(mod, Module):
                raise AttributeError(
                    f"'{item}' is not an nn.Module"
                )

        return mod

    def get_parameter(self, target: str) -> Parameter:
        """Return the parameter given by target path."""
        atoms = target.split('.')
        mod = self

        for item in atoms[:-1]:
            mod = getattr(mod, item)

        return getattr(mod, atoms[-1])

    def get_buffer(self, target: str) -> Tensor:
        """Return the buffer given by target path."""
        atoms = target.split('.')
        mod = self

        for item in atoms[:-1]:
            mod = getattr(mod, item)

        return getattr(mod, atoms[-1])

    def _get_name(self):
        """Return the class name."""
        return self.__class__.__name__

    @property
    def _backward_hooks(self):
        return OrderedDict()

    @property
    def _forward_hooks(self):
        return OrderedDict()

    @property
    def _forward_pre_hooks(self):
        return OrderedDict()

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False):
        """Register a forward hook (stub)."""
        return _RemovableHandle()

    def register_forward_pre_hook(self, hook, *, prepend=False, with_kwargs=False):
        """Register a forward pre-hook (stub)."""
        return _RemovableHandle()

    def register_backward_hook(self, hook):
        """Register a backward hook (stub)."""
        return _RemovableHandle()

    def register_full_backward_hook(self, hook, prepend=False):
        """Register a full backward hook (stub)."""
        return _RemovableHandle()

    def apply(self, fn):
        """Apply fn recursively to every submodule and self."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Load parameters from state_dict. Called by load_state_dict."""
        for name, param in self._parameters.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if param is not None:
                    param.data = input_param
            elif strict:
                missing_keys.append(key)

        for name, buf in self._buffers.items():
            key = prefix + name
            if key in state_dict:
                input_buf = state_dict[key]
                if buf is not None:
                    buf.data = input_buf
            elif strict:
                missing_keys.append(key)

    def _named_members(self, get_members_fn, prefix='', recurse=True, **kwargs):
        """Helper for named_parameters / named_buffers."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or id(v) in memo:
                    continue
                memo.add(id(v))
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v


class _IncompatibleKeys:
    """Container for incompatible keys from load_state_dict."""

    def __init__(self, missing_keys, unexpected_keys):
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys

    def __repr__(self):
        return f'_IncompatibleKeys(missing_keys={self.missing_keys}, unexpected_keys={self.unexpected_keys})'


class _RemovableHandle:
    """Handle to remove a hook."""

    def __init__(self):
        pass

    def remove(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()
