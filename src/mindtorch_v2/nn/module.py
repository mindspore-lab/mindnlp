from __future__ import annotations

from collections import OrderedDict

from .parameter import Parameter


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._buffers = {}
        self._non_persistent_buffers_set = set()
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def register_buffer(self, name, tensor, persistent=True):
        self._buffers[name] = tensor
        if not persistent:
            self._non_persistent_buffers_set.add(name)
        if tensor is not None:
            super().__setattr__(name, tensor)

    def register_parameter(self, name, param):
        self._parameters[name] = param
        if param is not None:
            super().__setattr__(name, param)

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix='', recurse=True):
        yield from self._named_members(
            lambda m: m._parameters.items(), prefix=prefix, recurse=recurse
        )

    def buffers(self, recurse=True):
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix='', recurse=True):
        yield from self._named_members(
            lambda m: m._buffers.items(), prefix=prefix, recurse=recurse
        )

    def children(self):
        for module in self._modules.values():
            yield module

    def named_children(self):
        for name, module in self._modules.items():
            yield name, module

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                yield from module.named_modules(memo, submodule_prefix)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf if keep_vars else (buf.detach() if hasattr(buf, 'detach') else buf)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.', keep_vars)
        return destination

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        device = kwargs.get('device', None)
        dtype = kwargs.get('dtype', None)
        if len(args) == 1:
            if isinstance(args[0], str):
                device = args[0]
            else:
                dtype = args[0]
        def convert(t):
            return t.to(*args, **kwargs)
        self._apply(convert)
        return self

    def cpu(self):
        return self.to(device='cpu')

    def float(self):
        from .. import float32
        return self.to(dtype=float32)

    def half(self):
        from .. import float16
        return self.to(dtype=float16)

    def double(self):
        return self

    def bfloat16(self):
        return self

    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def zero_grad(self, set_to_none=True):
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def extra_repr(self):
        return ''

    def __repr__(self):
        extra = self.extra_repr()
        extra_lines = extra.split('\n') if extra else []
        child_lines = []
        for key, module in self._modules.items():
            child_lines.append(f'({key}): {repr(module)}')
        lines = extra_lines + child_lines
        main_str = self._get_name() + '('
        if lines:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        keys = module_attrs + attrs
        keys += list(self._parameters.keys())
        keys += list(self._modules.keys())
        keys += list(self._buffers.keys())
        return sorted(set(keys))

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
                        new_value._skip_init = True  # Mark tensor to skip reinitialization
                        module._parameters[param_name] = new_value
                        if param_name in module.__dict__:
                            module.__dict__[param_name] = new_value
                    else:
                        param.data = new_value
                        param._skip_init = True  # Mark tensor to skip reinitialization
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
                new_param = fn(param)
                self._parameters[key] = new_param
                super().__setattr__(key, new_param)

        for key, buf in self._buffers.items():
            if buf is not None:
                new_buf = fn(buf)
                self._buffers[key] = new_buf
                super().__setattr__(key, new_buf)

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
