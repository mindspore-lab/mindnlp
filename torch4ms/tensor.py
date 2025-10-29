import torch
import torch.utils._python_dispatch as torch_dispatch

import mindspore
from torch4ms.aten.utils import ms2pt_dtype


class Tensor(torch.Tensor):

    @staticmethod
    def __new__(cls, data, requires_grad=False):
        dtype = ms2pt_dtype(data.dtype)
        shape = data.shape

        if dtype is None:
            dtype = torch.float32

        if not (dtype.is_floating_point or dtype.is_complex):
            requires_grad = False

        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device="meta",
            requires_grad=requires_grad,
        )

    def __init__(self, data: mindspore.Tensor, requires_grad=False):
        super().__init__()
        self._data = data

    def __str__(self):
        return "Tensor({})".format(self._data)

    __repr__ = __str__

    @property
    def shape(self):
        return torch.Size(self._data.shape)

    @property
    def ndim(self):
        return self._data.ndim

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # TODO(hanq): figure out why is dispatch mode not sufficient
        if func == torch.ops.prim.device.default:
            return torch.device("privateuseone", 0)
        raise AssertionError(
            "torchax Tensors can only do math within the torchax environment."
            "Please wrap your code with `with torchax.default_env()` or "
            "call torchax.enable_globally() before."
        )

    def numpy(self):
        return self._data.asnumpy()

    def mindspore(self):
        return self._data

    @property
    def dtype(self):
        return ms2pt_dtype(self._data.dtype)

    def dim(self):
        return self.ndim

    @property
    def device(self):
        return torch.device("ms:0")

    @property
    def ms_device(self):
        return self._data.device

    def tolist(self):
        return self._data.tolist()


class MSFunctionMode(torch.overrides.TorchFunctionMode):
    """Context manager that dispatches torch function calls to MindSpore."""

    def __torch_function__(self, func, types, args=(), kwargs=None) -> torch.Tensor:
        try:
            return dispatch(func, types, args, kwargs)
        except Exception:
            pass
        return func(*args, **(kwargs or {}))


class MSDispatchMode(torch_dispatch.TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(func, types, args, kwargs)
        if isinstance(func, torch._ops.OpOverloadPacket):
            with self:
                return func(*args, **kwargs)
        # Only functions under these namespaces will be intercepted
        if func.namespace not in (
            "aten",
            "_c10d_functional",
            "torchvision",
            "xla",
        ):
            return func(*args, **kwargs)
        return dispatch(func, types, args, kwargs)


def dispatch(func, types, args, kwargs):
    kwargs = kwargs or {}
    # if func in TENSOR_CONSTRUCTORS:
    #     return self._handle_tensor_constructor(func, args, kwargs)
    if func in (
        torch.Tensor.to,
        torch.ops.aten.lift_fresh.default,
        torch.ops.aten._to_copy,
        torch.ops.aten._to_copy.default,
    ):
        print(func, types, args, kwargs)
        return self._torch_Tensor_to(args, kwargs)

    # If the func doesn't act on Tensor, and is not a tensor constructor,
    # We should skip and let torch handle it.

    tensor_args = [
        t for t in torch_pytree.tree_flatten(args)[0] if isinstance(t, torch.Tensor)
    ]

    def is_not_torchax_tensor(x):
        return not isinstance(x, Tensor) and not isinstance(x, View)

    if tensor_args and all(is_not_torchax_tensor(t) for t in tensor_args):
        res = func(*args, **kwargs)
        return res

    with jax.named_scope(_name_of_func(func)):
        op = self._get_op_or_decomp(func)

        old_args, old_kwargs = args, kwargs
        with self._dispatch_mode:
            args, kwargs = torch_pytree.tree_map_only(
                torch.distributed._functional_collectives.AsyncCollectiveTensor,
                torch.distributed._functional_collectives.wait_tensor,
                (args, kwargs),
            )

        try:
            if not op.is_view_op:
                args, kwargs = self.v2t_iso((args, kwargs))

            with self:
                if self.param.autocast_dtype is not None:
                    autocast_policy = amp.autocast_policy.get(func)
                    if autocast_policy is not None:
                        args, kwargs = amp.execute_policy(
                            autocast_policy, args, kwargs, self.param.autocast_dtype
                        )

            if op.is_jax_function:
                args, kwargs = self.t2j_iso((args, kwargs))
        except AssertionError:
            if self.config.debug_mixed_tensor:
                breakpoint()
            else:
                raise

        if op.needs_env:
            kwargs["env"] = self

        if op.is_jax_function:
            res = op.func(*args, **kwargs)
        else:
            # enable dispatch mode because this op could be a composite autograd op
            # meaning, it will decompose in C++
            with self._dispatch_mode:
                res = op.func(*args, **kwargs)

        if op.is_jax_function:
            res = self.j2t_iso(res)

        if self.config.force_materialize_views and isinstance(res, View):
            res = res.torch()

        if self.config.debug_accuracy_for_each_op:
            debug_accuracy(func, old_args, old_kwargs, res)
        return res
