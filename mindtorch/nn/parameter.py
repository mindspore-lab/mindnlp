"""new Parameter"""
import uuid
import copy
from typing import Any
from mindspore import Tensor
from mindspore._c_expression import ParamInfo # pylint: disable=no-name-in-module
from mindspore.common._stub_tensor import StubTensor

import mindtorch

class Parameter(Tensor):
    grad = None
    _grad_fn = None

    def __init__(self, input_data=None, requires_grad=True, **kwargs):
        super().__init__(input_data)
        self._device = input_data._device
        self.meta = False
        self.param_info = ParamInfo()
        self.param_info.name = str(uuid.uuid4())
        self.param_info.parameter_shape = self._shape
        self.param_info.requires_grad = requires_grad
        self._requires_grad = requires_grad

    def __deepcopy__(self, memodict):
        new_obj = Parameter(self)
        new_obj._device = self.device
        return new_obj

    def __parameter__(self): # only for O2
        """For parse check."""

    @property
    def name(self): # only for O2
        """
        Get the name of the parameter.

        Examples:
            >>> from mindspore import Tensor, Parameter
            >>> import numpy as np
            >>> x = Parameter(Tensor(np.array([1, 2], dtype=np.float32)), name="param")
            >>> x.name = "param1"
            >>> x.name
            'param1'
        """
        return self.param_info.name


class UninitializedTensorMixin:
    _allowed_methods = [
        mindtorch.Tensor.__hash__,
        mindtorch.Tensor.size,
        mindtorch.Tensor.copy_,
        mindtorch.Tensor.is_complex,
        mindtorch.Tensor.is_floating_point,
        mindtorch.Tensor.half,
        mindtorch.Tensor.float,
        mindtorch.Tensor.double,
        mindtorch.Tensor.char,
        mindtorch.Tensor.short,
        mindtorch.Tensor.int,
        mindtorch.Tensor.long,
        mindtorch.Tensor.cuda,
        mindtorch.Tensor.cpu,
        mindtorch.Tensor.to,
        mindtorch.Tensor.get_device,
        mindtorch._has_compatible_shallow_copy_type,
    ]

    def materialize(self, shape, device=None, dtype=None):
        r"""Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`mindtorch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`mindtorch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        self.data = mindtorch.empty(shape, device=device, dtype=dtype)
        self.__class__ = self.cls_to_become

    @property
    def shape(self):
        raise RuntimeError(
            "Can't access the shape of an uninitialized parameter or buffer. "
            "This error usually happens in `load_state_dict` when trying to load "
            "an uninitialized parameter into an initialized one. "
            "Call `forward` to initialize the parameters before accessing their attributes."
        )

    def share_memory_(self):
        raise RuntimeError(
            "Can't share memory on an uninitialized parameter or buffer. "
            "Call `forward` to initialize the parameters before calling "
            "`module.share_memory()`."
        )

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (self.__class__, (self.requires_grad,))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # method-wrapper is to detect access to Tensor properties that are
        # wrapped in descriptors
        if func in cls._allowed_methods or func.__class__.__name__ == "method-wrapper":
            if kwargs is None:
                kwargs = {}
            return super().__torch_function__(func, types, args, kwargs)
        raise ValueError(
            f"Attempted to use an uninitialized parameter in {func}. "
            "This error happens when you are using a `LazyModule` or "
            f"explicitly manipulating `mindtorch.nn.parameter.{cls.__name__}` "
            "objects. When using LazyModules Call `forward` with a dummy batch "
            "to initialize the parameters before calling torch functions"
        )

class UninitializedParameter(UninitializedTensorMixin, Parameter):
    r"""A parameter that is not initialized.

    Uninitialized Parameters are a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = Parameter

    def __new__(cls, requires_grad=True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        data = mindtorch.empty(0, **factory_kwargs)
        return mindtorch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.requires_grad, self.data.device, self.data.dtype)
            memo[id(self)] = result
            return result


def is_lazy(param: Any) -> bool:
    """
    Returns whether ``param`` is an ``UninitializedParameter`` or ``UninitializedBuffer``.

    Args:
        param (Any): the input to check.
    """
    return isinstance(param, UninitializedTensorMixin)


class Buffer(Tensor):
    r"""A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

    Buffers are :class:`~mindtorch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s -- when they're
    assigned as Module attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~mindtorch.nn.Module.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~mindtorch.nn.Module.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            data = mindtorch.empty(0)

        t = data.detach().requires_grad_(data.requires_grad)
        t.persistent = persistent
        t._is_buffer = True
        return t

class UninitializedBuffer(UninitializedTensorMixin, mindtorch.Tensor):
    r"""A buffer that is not initialized.

    Uninitialized Buffer is a a special case of :class:`mindtorch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`mindtorch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`mindtorch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """

    cls_to_become = mindtorch.Tensor

    def __new__(
        cls, requires_grad=False, device=None, dtype=None, persistent=True
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        data = mindtorch.empty(0, **factory_kwargs)
        ret = mindtorch.Tensor._make_subclass(cls, data, requires_grad)
        ret.persistent = persistent
        ret._is_buffer = True
        return ret
