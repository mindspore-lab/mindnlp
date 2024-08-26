"""Module"""
import warnings
from typing import Dict, Optional, Callable, Set, overload, TypeVar, Any, Iterator, Tuple, Union, \
    Mapping, List
import itertools
from collections import OrderedDict, namedtuple

import mindspore
from mindspore import Tensor, Parameter
from mindspore.common._stub_tensor import StubTensor

from ...utils import hooks
from ...utils.hooks import RemovableHandle


T = TypeVar('T', bound='Module')

class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

_EXTRA_STATE_KEY_SUFFIX = '_extra_state'

_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()


_global_backward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_backward_hooks: Dict[int, Callable] = OrderedDict()
_global_is_full_backward_hook: Optional[bool] = None
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()


class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import minispore.nn as nn
        import minispore.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))
    """

    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    _non_persistent_buffers_set: Set[str]
    _backward_pre_hooks: Dict[int, Callable]
    _backward_hooks: Dict[int, Callable]
    _is_full_backward_hook: Optional[bool]
    _forward_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_hooks_with_kwargs: Dict[int, bool]
    # forward hooks that should always be called even if an exception is raised
    _forward_hooks_always_called: Dict[int, bool]
    _forward_pre_hooks: Dict[int, Callable]
    # Marks whether the corresponding _forward_hooks accept kwargs or not.
    # As JIT does not support Set[int], this dict is used as a set, where all
    # hooks represented in this dict accept kwargs.
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
    _load_state_dict_post_hooks: Dict[int, Callable]
    _modules: Dict[str, Optional['Module']]
    call_super_init: bool = False
    _compiled_call_impl : Optional[Callable] = None

    def __init__(self):
        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_buffers', OrderedDict())
        super().__setattr__('_non_persistent_buffers_set', set())
        super().__setattr__('_backward_pre_hooks', OrderedDict())
        super().__setattr__('_backward_hooks', OrderedDict())
        super().__setattr__('_is_full_backward_hook', None)
        super().__setattr__('_forward_hooks', OrderedDict())
        super().__setattr__('_forward_hooks_with_kwargs', OrderedDict())
        super().__setattr__('_forward_hooks_always_called', OrderedDict())
        super().__setattr__('_forward_pre_hooks', OrderedDict())
        super().__setattr__('_forward_pre_hooks_with_kwargs', OrderedDict())
        super().__setattr__('_state_dict_hooks', OrderedDict())
        super().__setattr__('_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_pre_hooks', OrderedDict())
        super().__setattr__('_load_state_dict_post_hooks', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

    def forward(self, *input, **kwargs):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {type(name)}")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, (Tensor, mindspore.Tensor)):
            raise TypeError(f"cannot assign '{type(tensor)}' object to buffer '{name}' "
                            "(torch Tensor or None required)"
                            )
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, tensor)
                if output is not None:
                    tensor = output
            if isinstance(tensor, StubTensor):
                tensor = tensor.stub_sync()
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {type(name)}")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param)}' object to parameter '{name}' "
                            "(torch.nn.Parameter or None required)"
                            )
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def add_module(self, name, module):
        """Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            parameter (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(type(module)))
        if hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        self._modules[name] = module

    def get_extra_state(self) -> Any:
        """Return any extra state to include in the module's state_dict.

        Implement this and a corresponding :func:`set_extra_state` for your module
        if you need to store extra state. This function is called when building the
        module's `state_dict()`.

        Note that extra state should be picklable to ensure working serialization
        of the state_dict. We only provide provide backwards compatibility guarantees
        for serializing Tensors; other objects may break backwards compatibility if
        their serialized pickled form changes.

        Returns:
            object: Any extra state to store in the module's state_dict
        """
        raise RuntimeError(
            "Reached a code path in Module.get_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug.")


    def set_extra_state(self, state: Any) -> None:
        """Set extra state contained in the loaded `state_dict`.

        This function is called from :func:`load_state_dict` to handle any extra state
        found within the `state_dict`. Implement this function and a corresponding
        :func:`get_extra_state` for your module if you need to store extra state within its
        `state_dict`.

        Args:
            state (dict): Extra state from the `state_dict`
        """
        raise RuntimeError(
            "Reached a code path in Module.set_extra_state() that should never be called. "
            "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
            "to report this bug.")

    def _apply(self, fn, recurse=True):
        if recurse:
            for module in self.children():
                module._apply(fn)

        for key, param in self._parameters.items():
            if param is None:
                continue
            param_applied = fn(param)

            assert isinstance(param, Parameter)
            out_param = Parameter(param_applied, requires_grad=param.requires_grad)
            self._parameters[key] = out_param

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def apply(self, fn):
        """Applies ``fn`` recursively to every submodule (as returned by ``.children()``)
        as well as self. Typical use includes initializing the parameters of a model
        (see also :ref:`torch-nn-init`).

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example:
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.data.fill_(1.0)
            >>>         print(m.weight)
            >>>
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear (2 -> 2)
            Parameter containing:
             1  1
             1  1
            [torch.FloatTensor of size 2x2]
            Linear (2 -> 2)
            Parameter containing:
             1  1
             1  1
            [torch.FloatTensor of size 2x2]
            Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            )
        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def _wrapped_call_impl(self, *args, **kwargs):
        return self._call_impl(*args, **kwargs)

    # torchrec tests the code consistency with the following code
    # fmt: off
    def _call_impl(self, *args, **kwargs):
        forward_call = self.forward
        # If we don't have any hooks, we want to skip the rest of the logic in
        # this function, and just call forward.
        if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
                or _global_backward_pre_hooks or _global_backward_hooks
                or _global_forward_hooks or _global_forward_pre_hooks):
            return forward_call(*args, **kwargs)

        try:
            result = None
            called_always_called_hooks = set()

            full_backward_hooks, non_full_backward_hooks = [], []
            backward_pre_hooks = []
            if self._backward_pre_hooks or _global_backward_pre_hooks:
                backward_pre_hooks = self._get_backward_pre_hooks()

            if self._backward_hooks or _global_backward_hooks:
                full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

            if _global_forward_pre_hooks or self._forward_pre_hooks:
                for hook_id, hook in (
                    *_global_forward_pre_hooks.items(),
                    *self._forward_pre_hooks.items(),
                ):
                    if hook_id in self._forward_pre_hooks_with_kwargs:
                        args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
                        if args_kwargs_result is not None:
                            if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
                                args, kwargs = args_kwargs_result
                            else:
                                raise RuntimeError(
                                    "forward pre-hook must return None or a tuple "
                                    f"of (new_args, new_kwargs), but got {args_kwargs_result}."
                                )
                    else:
                        args_result = hook(self, args)
                        if args_result is not None:
                            if not isinstance(args_result, tuple):
                                args_result = (args_result,)
                            args = args_result

            bw_hook = None
            # if full_backward_hooks or backward_pre_hooks:
            #     bw_hook = BackwardHook(self, full_backward_hooks, backward_pre_hooks)
            #     args = bw_hook.setup_input_hook(args)

            result = forward_call(*args, **kwargs)
            if _global_forward_hooks or self._forward_hooks:
                for hook_id, hook in (
                    *_global_forward_hooks.items(),
                    *self._forward_hooks.items(),
                ):
                    # mark that always called hook is run
                    if hook_id in self._forward_hooks_always_called or hook_id in _global_forward_hooks_always_called:
                        called_always_called_hooks.add(hook_id)

                    if hook_id in self._forward_hooks_with_kwargs:
                        hook_result = hook(self, args, kwargs, result)
                    else:
                        hook_result = hook(self, args, result)

                    if hook_result is not None:
                        result = hook_result

            if bw_hook:
                if not isinstance(result, (mindspore.Tensor, tuple)):
                    warnings.warn("For backward hooks to be called,"
                                  " module output should be a Tensor or a tuple of Tensors"
                                  f" but received {type(result)}")
                result = bw_hook.setup_output_hook(result)

            # Handle the non-full backward hooks
            if non_full_backward_hooks:
                var = result
                while not isinstance(var, mindspore.Tensor):
                    if isinstance(var, dict):
                        var = next(v for v in var.values() if isinstance(v, mindspore.Tensor))
                    else:
                        var = var[0]
                # grad_fn = var.grad_fn
                # if grad_fn is not None:
                #     for hook in non_full_backward_hooks:
                #         grad_fn.register_hook(_WrappedHook(hook, self))
                #     self._maybe_warn_non_full_backward_hook(args, result, grad_fn)

            return result

        except Exception:
            # run always called hooks if they have not already been run
            # For now only forward hooks have the always_call option but perhaps
            # this functionality should be added to full backward hooks as well.
            for hook_id, hook in _global_forward_hooks.items():
                if hook_id in _global_forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                    try:
                        hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn("global module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue

            for hook_id, hook in self._forward_hooks.items():
                if hook_id in self._forward_hooks_always_called and hook_id not in called_always_called_hooks:  # type: ignore[possibly-undefined]
                    try:
                        if hook_id in self._forward_hooks_with_kwargs:
                            hook_result = hook(self, args, kwargs, result)  # type: ignore[possibly-undefined]
                        else:
                            hook_result = hook(self, args, result)  # type: ignore[possibly-undefined]
                        if hook_result is not None:
                            result = hook_result
                    except Exception as e:
                        warnings.warn("module forward hook with ``always_call=True`` raised an exception "
                                      f"that was silenced as another error was raised in forward: {str(e)}")
                        continue
            # raise exception raised in try block
            raise
    # fmt: on

    __call__: Callable[..., Any] = _wrapped_call_impl

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_compiled_call_impl", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Support loading old checkpoints that don't have the following attrs:
        if "_forward_pre_hooks" not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if "_forward_pre_hooks_with_kwargs" not in self.__dict__:
            self._forward_pre_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_with_kwargs" not in self.__dict__:
            self._forward_hooks_with_kwargs = OrderedDict()
        if "_forward_hooks_always_called" not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()
        if "_state_dict_hooks" not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if "_state_dict_pre_hooks" not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_pre_hooks" not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_post_hooks" not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if "_non_persistent_buffers_set" not in self.__dict__:
            self._non_persistent_buffers_set = set()
        if "_is_full_backward_hook" not in self.__dict__:
            self._is_full_backward_hook = None
        if "_backward_pre_hooks" not in self.__dict__:
            self._backward_pre_hooks = OrderedDict()

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign '{type(value)}' as parameter '{name}' "
                                "(mindspore.Parameter or None expected)"
                                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign '{type(value)}' as child module '{name}' "
                                    "(torch.nn.Module or None expected)"
                                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, Tensor):
                        raise TypeError(f"cannot assign '{type(value)}' as buffer '{name}' "
                                        "(torch.Tensor or None expected)"
                                        )
                    for hook in _global_buffer_registration_hooks.values():
                        output = hook(self, name, value)
                        if output is not None:
                            value = output
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)


    def extra_repr(self) -> str:
        r"""Set the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''


    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def cuda(self):
        return self

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        r"""Copy parameters and buffers from :attr:`state_dict` into only this module, but not its descendants.

        This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        Additionally, :attr:`local_metadata` can also contain the key
        `assign_to_params_buffers` that indicates whether keys should be
        assigned their corresponding tensor in the state_dict.

        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in self._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not isinstance(input_param, Tensor):
                    error_msgs.append(f'While copying the parameter named "{key}", '
                                      'expected torch.Tensor or Tensor-like object from checkpoint but '
                                      f'received {type(input_param)}'
                                      )
                    continue

                # This is used to avoid copying uninitialized parameters into
                # non-lazy modules, since they dont have the hook to do the checks
                # in such case, it will error when accessing the .shape attribute.

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append(f'size mismatch for {key}: copying a param with shape {input_param.shape} from checkpoint, '
                                      f'the shape in current model is {param.shape}.')
                    continue


                try:
                    if assign_to_params_buffers:
                        # Shape checks are already done above
                        if (isinstance(param, Parameter)):
                            if not isinstance(input_param, Parameter):
                                input_param = Parameter(input_param, requires_grad=param.requires_grad)
                            else:
                                input_param.requires_grad = param.requires_grad
                        setattr(self, name, input_param)
                    else:
                        dtype = param.dtype
                        param.assign_value(input_param)
                        param.set_dtype(dtype)
                except Exception as ex:
                    action = "copying"
                    error_msgs.append(f'While {action} the parameter named "{key}", '
                                      f'whose dimensions in the model are {param.size()} and '
                                      f'whose dimensions in the checkpoint are {input_param.size()}, '
                                      f'an exception occurred : {ex.args}.'
                                      )
            elif strict:
                missing_keys.append(key)

        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "set_extra_state", Module.set_extra_state) is not Module.set_extra_state:
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
            elif strict:
                missing_keys.append(extra_state_key)
        elif strict and (extra_state_key in state_dict):
            unexpected_keys.append(extra_state_key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix) and key != extra_state_key:
                    input_name = key[len(prefix):].split(".", 1)
                    # Must be Module if it have attributes
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~torch.nn.Module.state_dict` function.

        .. warning::
            If :attr:`assign` is ``True`` the optimizer must be created after
            the call to :attr:`load_state_dict` unless
            :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
            assign (bool, optional): When ``False``, the properties of the tensors
                in the current module are preserved while when ``True``, the
                properties of the Tensors in the state dict are preserved. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                for which the value from the module is preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = OrderedDict(state_dict)

        if metadata is not None:
            # mypy isn't aware that "_metadata" exists in state_dict
            state_dict._metadata = metadata  # type: ignore[attr-defined]

        def load(module, local_state_dict, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            if assign:
                local_metadata['assign_to_params_buffers'] = assign
            module._load_from_state_dict(
                local_state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + '.'
                    child_state_dict = {k: v for k, v in local_state_dict.items() if k.startswith(child_prefix)}
                    load(child, child_state_dict, child_prefix)  # noqa: F821

            # Note that the hook can modify missing_keys and unexpected_keys.
            incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    "Hooks registered with ``register_load_state_dict_post_hook`` are not"
                    "expected to return new values, if incompatible_keys need to be modified,"
                    "it should be done inplace."
                )

        load(self, state_dict)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join(f'"{k}"' for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)


    def _named_members(self, get_members_fn, prefix='', recurse=True, remove_duplicate: bool = True):
        r"""Help yield various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for param in model.parameters():
            >>>     print(type(param), param.shape)
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

    def get_parameters(self, expand=True):
        return self.parameters(expand)

    def named_parameters(
            self,
            prefix: str = '',
            recurse: bool = True,
            remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.shape)

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen

    def parameters_and_names(self, name_prefix='', expand=True):
        return self.named_parameters(name_prefix, expand)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.shape)
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf


    def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool, optional): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

        Yields:
            (str, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.shape)

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen

    def _all_buffers(self, memo=None):
        if memo is None:
            memo = set()
        for name, b in self._buffers.items():
            if b is not None and b not in memo:
                memo.add(b)
                yield b
        for module in self.children():
            for b in module._all_buffers(memo):
                yield b

    def children(self):
        """Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example:
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self):
        """Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            >>>     print(idx, '->', m)
            0 -> Sequential (
              (0): Linear (2 -> 2)
              (1): Linear (2 -> 2)
            )
            1 -> Linear (2 -> 2)
        """
        for name, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (str, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
            ...     print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                yield from module.named_modules(memo, submodule_prefix, remove_duplicate)

    def cells_and_names(self, cells=None, name_prefix=''):
        return self.named_modules(cells, name_prefix)

    def train(self, mode=True):
        """Sets the module in training mode.

        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    set_train = train

    def eval(self):
        """Sets the module in evaluation mode.

        This has any effect only on modules such as Dropout or BatchNorm.
        """
        return self.train(False)

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self


    def _get_name(self):
        return self.__class__.__name__

    def to(self, *args, **kwargs):
        return self

    def float(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``float`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.float() if t.is_floating_point() else t)


    def double(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``double`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.double() if t.is_floating_point() else t)


    def half(self: T) -> T:
        r"""Casts all floating point parameters and buffers to ``half`` datatype.

        .. note::
            This method modifies the module in-place.

        Returns:
            Module: self
        """
        return self._apply(lambda t: t.half() if t.is_floating_point() else t)


    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Save module state to the `destination` dictionary.

        The `destination` dictionary will contain the state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buf
        extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
        if getattr(self.__class__, "get_extra_state", Module.get_extra_state) is not Module.get_extra_state:
            destination[extra_state_key] = self.get_extra_state()

    # The user can pass an optional arbitrary mappable object to `state_dict`, in which case `state_dict` returns
    # back that same object. But if they pass nothing, an `OrderedDict` is created and returned.
    T_destination = TypeVar('T_destination', bound=Dict[str, Any])

    @overload
    def state_dict(self, *, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination:
        ...

    @overload
    def state_dict(self, *, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]:
        ...

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        r"""Return a dictionary containing references to the whole state of the module.

        Both parameters and persistent buffers (e.g. running averages) are
        included. Keys are corresponding parameter and buffer names.
        Parameters and buffers set to ``None`` are not included.

        .. note::
            The returned object is a shallow copy. It contains references
            to the module's parameters and buffers.

        .. warning::
            Currently ``state_dict()`` also accepts positional arguments for
            ``destination``, ``prefix`` and ``keep_vars`` in order. However,
            this is being deprecated and keyword arguments will be enforced in
            future releases.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            prefix (str, optional): a prefix added to parameter and buffer
                names to compose the keys in state_dict. Default: ``''``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> module.state_dict().keys()
            ['bias', 'weight']

        """
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = {}
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination

    def parameters_dict(self, recurse=True):
        param_dict = OrderedDict()
        for name, param in self.named_parameters(recurse=recurse, remove_duplicate=False):
            param_dict[name] = param
        return param_dict

    def register_forward_pre_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any]], Optional[Tuple[Any, Dict[str, Any]]]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`torch.nn.modules.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``
            with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
                given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(
            self._forward_pre_hooks,
            extra_dict=self._forward_pre_hooks_with_kwargs
        )
        self._forward_pre_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_pre_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle


    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.

        The hook will be called every time after :func:`forward` has computed an output.

        If ``with_kwargs`` is ``False`` or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        output. It can modify the input inplace but it will not have effect on
        forward since this is called after :func:`forward` is called. The hook
        should have the following signature::

            hook(module, args, output) -> None or modified output

        If ``with_kwargs`` is ``True``, the forward hook will be passed the
        ``kwargs`` given to the forward function and be expected to return the
        output possibly modified. The hook should have the following signature::

            hook(module, args, kwargs, output) -> None or modified output

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`torch.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`torch.nn.modules.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            with_kwargs (bool): If ``True``, the ``hook`` will be passed the
                kwargs given to the forward function.
                Default: ``False``

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(
            self._forward_hooks,
            extra_dict=self._forward_hooks_with_kwargs
        )
        self._forward_hooks[handle.id] = hook
        if with_kwargs:
            self._forward_hooks_with_kwargs[handle.id] = True

        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle
