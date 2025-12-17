import abc
import logging
from collections import defaultdict, deque
from collections.abc import MutableMapping, Sequence
from typing import (Any, Callable, NamedTuple, Optional, Union, Iterator)
from typing_extensions import TypeAlias
from weakref import WeakKeyDictionary, WeakValueDictionary
from contextlib import contextmanager
import mindtorch
from mindtorch.utils.hooks import RemovableHandle
from mindspore._c_expression import run_backward

log = logging.getLogger(__name__)

__all__ = ["save_on_cpu", "GradientEdge", "get_gradient_edge"]


class saved_tensors_hooks:
    def __init__(
            self,
            pack_hook: Callable[[mindtorch.Tensor], Any],
            unpack_hook: Callable[[Any], mindtorch.Tensor],
    ) -> None:
        log.warning(f"{self.__class__.__name__} is a placeholder with empty __enter__ and __exit__ functions!")
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None: ...

    def __exit__(self, *args: object) -> None: ...


@contextmanager
def save_on_cpu(*arg, **kwarg):
    yield

class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def next_functions(self) -> tuple[tuple[Optional["Node"], int], ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _input_metadata(self) -> list[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: mindtorch.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        raise NotImplementedError

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        return NotImplemented

def _get_grad_fn_or_grad_acc(t: Union[mindtorch.Tensor, "GradientEdge"]) -> Node:
    if isinstance(t, GradientEdge):
        return t.node
    if t.requires_grad and t.grad_fn is None:
        with mindtorch.enable_grad():
            node = t.view_as(t).grad_fn.next_functions[0][0]  # type: ignore[union-attr]
    else:
        node = t.grad_fn
    assert node is not None
    return node

class GradientEdge(NamedTuple):
    node: Node
    output_nr: int

def get_gradient_edge(tensor: mindtorch.Tensor) -> GradientEdge:
    if not tensor.requires_grad:
        raise RuntimeError( "It is not possible to get the gradient edge for a Tensor that does not require gradients")
    grad_fn = _get_grad_fn_or_grad_acc(tensor)
    return GradientEdge(grad_fn, tensor.output_nr)

class _MultiHandle(RemovableHandle):
    handles: tuple[RemovableHandle, ...]
    def __init__(self, handles: tuple[RemovableHandle, ...]) -> None:
        self.handles = handles

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __getstate__(self) -> tuple[RemovableHandle, ...]:
        return self.handles

    def __setstate__(self, state: tuple[RemovableHandle, ...]) -> None:
        self.handles = state

_allow_mutation_on_saved_tensors_enabled: bool = False
_TID: TypeAlias = tuple[int, int, int]
_SID: TypeAlias = tuple[int, int]

def _get_tid(tensor: mindtorch.Tensor) -> _TID: # FIXME: This is almost definitely a bug.
    if isinstance(tensor, (mindtorch._subclasses.fake_tensor.FakeTensor, mindtorch._subclasses.functional_tensor.FunctionalTensor)):
        data_ptr = 0
    else:
        data_ptr = tensor.data_ptr()
    return (id(tensor), data_ptr, tensor._version)

def _get_sid(tensor: mindtorch.Tensor) -> _SID: # FIXME: This is almost definitely a bug.
    if isinstance(tensor, (mindtorch._subclasses.fake_tensor.FakeTensor, mindtorch._subclasses.functional_tensor.FunctionalTensor)):
        data_ptr = 0
    else:
        data_ptr = tensor.data_ptr()
    return (data_ptr, tensor._version)

class _Handle:
    pass

class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        def pack_hook(tensor: mindtorch.Tensor) -> _Handle:
            tid = _get_tid(tensor)
            sid = _get_sid(tensor)
            handle: Optional[_Handle] = None
            ctx.sid_to_tid[sid].add(tid)
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = tensor
            else:
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(handle: _Handle) -> mindtorch.Tensor:
            error_msg = ("Trying to backward outside of the 'allow_mutation_on_saved_tensors' context in which the graph was originally recorded.")
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res
        super().__init__(pack_hook, unpack_hook)

class _AllowMutationOnSavedContext:
    def __init__(self) -> None:
        self.cloned: MutableMapping[_Handle, mindtorch.Tensor] = WeakKeyDictionary()
        self.original: MutableMapping[_Handle, mindtorch.Tensor] = WeakKeyDictionary()
        self.tid_to_weakhandle: MutableMapping[_TID, _Handle] = WeakValueDictionary()
        self.sid_to_tid: dict[_SID, set[_TID]] = defaultdict(set)

    def clear(self) -> None:
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()

def _register_logging_hooks_on_whole_graph(
    t_outputs: Sequence[Union[mindtorch.Tensor, GradientEdge]],
) -> Callable[[], None]:
    grad_fns = list(map(_get_grad_fn_or_grad_acc, t_outputs))

    def iter_graph(roots: list[Node]) -> Iterator[Node]:
        if not roots:
            return
        seen: set[Node] = set()
        q: deque[Node] = deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)
        while q:
            node = q.popleft()
            for fn, _ in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)
            yield node

    def fmt(t: Optional[mindtorch.Tensor]) -> str:
        from mindtorch.utils._dtype_abbrs import dtype_abbrs
        if t is None:
            return "None"
        return f"{dtype_abbrs[t.dtype]}[{', '.join(map(str, t.shape))}]"

    def prehook(grad_outputs: Sequence[Optional[mindtorch.Tensor]]) -> None:
        node = mindtorch._C._current_autograd_node()
        grad_outputs_str = f"[{','.join(fmt(t) for t in grad_outputs)}]"
        log_str = f"Executing: {node} with grad_outputs: {grad_outputs_str}"
        log.debug(log_str)
    handles = [node.register_prehook(prehook) for node in iter_graph(grad_fns)]

    def unregister_hooks() -> None:
        for handle in handles:
            handle.remove()
    return unregister_hooks

def _engine_run_backward(
    t_outputs: Sequence[Union[mindtorch.Tensor, GradientEdge]],
    *args: Any,
    **kwargs: Any,
) -> tuple[mindtorch.Tensor, ...]:
    attach_logging_hooks = log.getEffectiveLevel() <= logging.DEBUG
    if attach_logging_hooks:
        unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    try:
        return run_backward( t_outputs, *args, **kwargs)  # Calls into the C++ engine to run the backward pass
    finally:
        if attach_logging_hooks:
            unregister_hooks()  # type: ignore[possibly-undefined]

def register_multi_grad_hook():
    pass