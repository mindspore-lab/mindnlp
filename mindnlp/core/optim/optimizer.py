# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
"""Base optimizer."""
import functools
import math
import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from itertools import chain
from typing import (
    Any,
    Callable,
    cast,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec, Self, TypeAlias

import mindspore
from mindspore import Parameter
from .. import ops
from ..utils import get_default_dtype, hooks
from ..utils.hooks import RemovableHandle

Args: TypeAlias = Tuple[Any, ...]
Kwargs: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Any]
TensorListList: TypeAlias = List[List[mindspore.Tensor]]


GlobalOptimizerPreHook: TypeAlias = Callable[
    ["Optimizer", Args, Kwargs], Optional[Tuple[Args, Kwargs]]
]
GlobalOptimizerPostHook: TypeAlias = Callable[["Optimizer", Args, Kwargs], None]

__all__ = [
    "Optimizer",
    "register_optimizer_step_pre_hook",
    "register_optimizer_step_post_hook",
]
_global_optimizer_pre_hooks: Dict[int, GlobalOptimizerPreHook] = OrderedDict()
_global_optimizer_post_hooks: Dict[int, GlobalOptimizerPostHook] = OrderedDict()
_foreach_supported_types = [mindspore.Tensor, Parameter]


class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self) -> str:
        return "<required parameter>"


required = _RequiredParameter()


def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    return x.item() if isinstance(x, mindspore.Tensor) else x


def _stack_if_compiling(x):
    return x


def _dispatch_sqrt(
    x: float,
):  # float annotation is needed because of torchscript type inference
    return math.sqrt(x)


def _disable_dynamo_if_unsupported(single_tensor_fn=None):
    # workaround for torchscript BC
    # it requires all called functions to be in the
    # global environment at the site at which the
    # maybe_fallback closure is created
    if single_tensor_fn:
        globals()[single_tensor_fn.__name__] = single_tensor_fn

    def wrapper(func):
        import inspect

        ps = inspect.signature(func).parameters
        has_state_steps = True
        try:
            state_steps_ind = list(ps.keys()).index("state_steps")
        except ValueError:
            has_state_steps = False

        # Today, there are cases where we stack state steps
        # and pass them as the value arg of foreach ops.
        # Having state steps on cuda as the value arg is not supported in eager,
        # but this only occurs in the rare case that the user explicitly deletes
        # the capturable flag. If capturable=True, this is not a problem.
        @functools.wraps(func)
        def maybe_fallback(*args, **kwargs):
            return func(*args, **kwargs)

        return maybe_fallback

    return wrapper



def _view_as_real(params, *state_and_grads):
    for i, p in enumerate(params):
        if ops.is_complex(p):
            params[i] = ops.view_as_real(params[i])
            for s in state_and_grads:
                s[i] = ops.view_as_real(s[i])


def _get_scalar_dtype(is_fused=None):
    if is_fused:
        return mindspore.float32
    return (
        mindspore.float64 if get_default_dtype() == mindspore.float64 else mindspore.float32
    )



def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def register_optimizer_step_post_hook(hook: GlobalOptimizerPostHook) -> RemovableHandle:
    r"""Register a post hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_post_hooks)
    _global_optimizer_post_hooks[handle.id] = hook
    return handle


ParamsT: TypeAlias = Union[Iterable[mindspore.Tensor], Iterable[Dict[str, Any]]]

_P = ParamSpec("_P")
R = TypeVar("R")
T = TypeVar("T")


class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`mindspore.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    OptimizerPreHook: TypeAlias = Callable[[Self, Args, Kwargs], Optional[Tuple[Args, Kwargs]]]  # type: ignore[misc]
    OptimizerPostHook: TypeAlias = Callable[[Self, Args, Kwargs], None]  # type: ignore[misc]

    _optimizer_step_pre_hooks: Dict[int, OptimizerPreHook]
    _optimizer_step_post_hooks: Dict[int, OptimizerPostHook]
    _optimizer_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'
    _optimizer_state_dict_post_hooks: 'OrderedDict[int, Callable[["Optimizer", StateDict], Optional[StateDict]]]'
    _optimizer_load_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer", StateDict], Optional[StateDict]]]'
    _optimizer_load_state_dict_post_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'

    def __init__(self, params: ParamsT, defaults: Dict[str, Any]) -> None:  # noqa: D107
        self.defaults = defaults
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

        if isinstance(params, mindspore.Tensor):
            raise TypeError(
                "params argument given to the optimizer should be "
                "an iterable of Tensors or dicts, but got " + type(params)
            )

        self.state: DefaultDict[mindspore.Tensor, Any] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(cast(dict, param_group))

        # Allows _cuda_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True

    def __getstate__(self) -> Dict[str, Any]:  # noqa: D105
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:  # noqa: D105
        self.__dict__.update(state)
        if "_optimizer_step_pre_hooks" not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if "_optimizer_step_post_hooks" not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        if "_optimizer_state_dict_pre_hooks" not in self.__dict__:
            self._optimizer_state_dict_pre_hooks = OrderedDict()
        if "_optimizer_state_dict_post_hooks" not in self.__dict__:
            self._optimizer_state_dict_post_hooks = OrderedDict()
        if "_optimizer_load_state_dict_pre_hooks" not in self.__dict__:
            self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        if "_optimizer_load_state_dict_post_hooks" not in self.__dict__:
            self._optimizer_load_state_dict_post_hooks = OrderedDict()
        self.defaults.setdefault("differentiable", False)

    def __repr__(self) -> str:  # noqa: D105
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    def _optimizer_step_code(self) -> None:
        """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
       .
        """

    def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle:
        r"""Register an optimizer step pre hook which will be called before optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_pre_hooks)
        self._optimizer_step_pre_hooks[handle.id] = hook
        return handle

    def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle:
        r"""Register an optimizer step post hook which will be called after optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_step_post_hooks)
        self._optimizer_step_post_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ) -> RemovableHandle:  # noqa: D101
        r"""Register a state dict pre-hook which will be called before :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.
        The hook will be called with argument ``self`` before calling ``state_dict`` on ``self``.
        The registered hook can be used to perform pre-processing before the ``state_dict``
        call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_state_dict_pre_hooks)
        self._optimizer_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_state_dict_post_hook(
        self,
        hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a state dict post-hook which will be called after :meth:`~torch.optim.Optimizer.state_dict` is called.

        It should have the following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The hook will be called with arguments ``self`` and ``state_dict`` after generating
        a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
        return a new one. The registered hook can be used to perform post-processing
        on the ``state_dict`` before it is returned.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_state_dict_post_hooks)
        self._optimizer_state_dict_post_hooks[handle.id] = hook
        if prepend:
            self._optimizer_state_dict_post_hooks.move_to_end(handle.id, last=False)
        return handle

    def state_dict(self) -> StateDict:
        r"""Return the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * ``state``: a Dict holding current optimization state. Its content
            differs between optimizer classes, but some common characteristics
            hold. For example, state is saved per parameter, and the parameter
            itself is NOT saved. ``state`` is a Dictionary mapping parameter ids
            to a Dict with state corresponding to each parameter.
        * ``param_groups``: a List containing all parameter groups where each
            parameter group is a Dict. Each parameter group contains metadata
            specific to the optimizer, such as learning rate and weight decay,
            as well as a List of parameter IDs of the parameters in the group.

        NOTE: The parameter IDs may look like indices but they are just IDs
        associating state with param_group. When loading from a state_dict,
        the optimizer will zip the param_group ``params`` (int IDs) and the
        optimizer ``param_groups`` (actual ``nn.Parameter`` s) in order to
        match state WITHOUT additional verification.

        A returned state dict might look something like:

        .. code-block:: text

            {
                'state': {
                    0: {'momentum_buffer': tensor(...), ...},
                    1: {'momentum_buffer': tensor(...), ...},
                    2: {'momentum_buffer': tensor(...), ...},
                    3: {'momentum_buffer': tensor(...), ...}
                },
                'param_groups': [
                    {
                        'lr': 0.01,
                        'weight_decay': 0,
                        ...
                        'params': [0]
                    },
                    {
                        'lr': 0.001,
                        'weight_decay': 0.5,
                        ...
                        'params': [1, 2, 3]
                    }
                ]
            }

        """
        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)

        # Save order indices instead of Tensors
        param_mappings: Dict[int, int] = {}
        start_index = 0

        def pack_group(group: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != "params"}
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed["params"] = [param_mappings[id(p)] for p in group["params"]]
            start_index += len(packed["params"])
            return packed

        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {
            (param_mappings[id(k)] if isinstance(k, mindspore.Tensor) else k): v
            for k, v in self.state.items()
        }

        state_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }

        for post_hook in self._optimizer_state_dict_post_hooks.values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        return state_dict

    @staticmethod
    def _process_value_according_to_param_policy(
        param: mindspore.Tensor,
        value: mindspore.Tensor,
        param_id: int,
        param_groups: List[Dict[Any, Any]],
        key: Hashable = None,
    ) -> mindspore.Tensor:
        # Floating-point types are a bit special here. They are the only ones
        # that are assumed to always match the type of params.
        # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
        # UNLESS fused or capturable, see note [special device hosting for step]
        fused = False
        capturable = False
        assert param_groups is not None
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg["fused"] if "fused" in pg else False
                capturable = pg["capturable"] if "capturable" in pg else False
                break
        if key == "step":
            if capturable or fused:
                return value.to(dtype=mindspore.float32)
            else:
                return value
        else:
            if param.is_floating_point():
                return value.to(dtype=param.dtype)
            else:
                return value

    def register_load_state_dict_pre_hook(
        self,
        hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
        prepend: bool = False,
    ) -> RemovableHandle:  # noqa: D205 D400
        r"""Register a load_state_dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The ``optimizer`` argument is the optimizer instance being used and the
        ``state_dict`` argument is a shallow copy of the ``state_dict`` the user
        passed in to ``load_state_dict``. The hook may modify the state_dict inplace
        or optionally return a new one. If a state_dict is returned, it will be used
        to be loaded into the optimizer.

        The hook will be called with argument ``self`` and ``state_dict`` before
        calling ``load_state_dict`` on ``self``. The registered hook can be used to
        perform pre-processing before the ``load_state_dict`` call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_pre_hooks)
        self._optimizer_load_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_load_state_dict_post_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ) -> RemovableHandle:  # noqa: D205 D400
        r"""Register a load_state_dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        The hook will be called with argument ``self`` after calling
        ``load_state_dict`` on ``self``. The registered hook can be used to
        perform post-processing after ``load_state_dict`` has loaded the
        ``state_dict``.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_post_hooks)
        self._optimizer_load_state_dict_post_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_post_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        return handle

    def load_state_dict(self, state_dict: StateDict) -> None:
        r"""Load the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, mindspore.Tensor):
                return Optimizer._process_value_according_to_param_policy(
                    param, value, param_id, param_groups, key
                )
            elif isinstance(value, dict):
                return {
                    k: _cast(
                        param, v, param_id=param_id, param_groups=param_groups, key=k
                    )
                    for k, v in value.items()
                }
            elif isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)  # type: ignore[call-arg]
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: DefaultDict[mindspore.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(
                    param, v, param_id=k, param_groups=state_dict["param_groups"]
                )
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(
            group: Dict[str, Any], new_group: Dict[str, Any]
        ) -> Dict[str, Any]:
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)


    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Perform a single optimization step to update parameter.

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        params = param_group["params"]
        if isinstance(params, mindspore.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, mindspore.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + type(param)
                )
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    f"parameter group didn't specify a value of required optimization parameter {name}"
                )
            else:
                param_group.setdefault(name, default)

        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; "
                "see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3,
            )

        param_set: Set[mindspore.Tensor] = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)
