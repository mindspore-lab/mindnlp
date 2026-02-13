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
