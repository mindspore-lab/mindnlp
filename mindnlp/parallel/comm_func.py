"""communication functional api."""
from mindspore import ops, Tensor
from mindspore._c_expression import Tensor as Tensor_ # pylint: disable=no-name-in-module
from mindspore.ops.operations._inner_ops import Send, Receive
from mindspore.communication import GlobalComm, get_group_rank_from_world_rank
from mindspore.ops._primitive_cache import _get_cache_prim

def isend(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0):
    """
    Send tensors to the specified dest_rank.

    Note:
        Send and Receive must be used in combination and have same tag.

    Args:
        tensor (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dst (int, optional): A required integer identifying the destination rank(global rank). Default: 0.
        group (str, optional): The communication group to work on.
            Default: "hccl_world_group" on Ascend, "nccl_world_group" on GPU.
        tag (int, optional): A required integer identifying the send/recv message tag. The message will
            be received by the Receive op with the same "tag". Default: 0.

    Raises:
        TypeError: `dst` is not an int or `group` is not a str。
        ValueError: If the rank ID of the process is greater than the rank size of the communication group.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import isend
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>>
        >>> init()
        >>> input_ = Tensor(np.ones([2, 8]).astype(np.float32))
        >>> isend(input_, 0)
    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For isend, the input tensor must be tensor")
    _dst = get_group_rank_from_world_rank(dst, group)
    _op = _get_cache_prim(Send)(tag, _dst, group, group)
    _depend = _get_cache_prim(ops.Depend)()
    return _depend(tensor, _op(tensor))


def irecv(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0):
    """
    Receive tensors from src.

    Note:
        Send and Receive must be used in combination and have same tag.
        The shape and dtype of input `tensor` is used to receive tensor, but the value
        of input `tensor` would not take effect.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. The shape and dtype of this
            tensor is used to receive tensor, but the value of input `tensor` would not take effect.
        src (int, optional): A required integer identifying the source rank(global rank). Default: 0.
        group (str, optional): The communication group to work on.
            Default: "hccl_world_group" on Ascend, "nccl_world_group" on GPU.
        tag (int, optional): A required integer identifying the send/recv message tag. The message will
            be received by the Send op with the same "tag". Default: 0.

    Returns:
        Tensor, the shape of output is :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If `src` is not an int or `group` is not a str.
        ValueError: If the rank ID of the process is greater than the rank size of the communication group.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> from mindspore import ops
        >>> import mindspore.nn as nn
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import irecv
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>>
        # Launch 2 processes.
        Process 0 send the following array to Process 1
        [[ 0.  1.]
         [ 2.  3.]]
        >>> init()
        >>> x = ms.Tensor(np.zeros([2, 2]))
        # Process 1 receive tensor from Process 0.
        >>> out = irecv(x, src=0)
        >>> print(out)
        [[ 0.  1.]
         [ 2.  3.]]
    """
    _src = get_group_rank_from_world_rank(src, group)
    shape = tensor.shape
    dtype = tensor.dtype
    _op = _get_cache_prim(Receive)(tag, _src, shape, dtype, group, group)
    return _op(tensor)

def broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP):
    """
    Broadcasts the tensor to the whole group.

    Note:
        The tensors must have the same shape and format in all processes of the collection.
        Only support PyNative mode, Graph mode is not currently supported.

    Args:
        tensor (Tensor): The tensor to be broadcasted. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        src (int, optional): Specifies the rank(global rank) of the process that broadcast the tensor.
            And only process `src` will broadcast the tensor.
        group (str, optional): The communication group to work on. Default: ``GlobalComm.WORLD_COMM_GROUP``.

    Returns:
        Tensor, tensor has the same shape as input tensor :math:`(x_1, x_2, ..., x_R)`.

    Raises:
        TypeError: If src is not an integer or group is not a string.
        RuntimeError: If device target is invalid, or backend is invalid, or distributed initialization fails.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        .. note::
            Before running the following examples, you need to configure the communication environment variables.

            For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
            without any third-party or configuration file dependencies.
            Please see the `msrun start up
            <https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.1/parallel/msrun_launcher.html>`_
            for more details.

            This example should be run with 2 devices.

        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindspore.communication import init
        >>> from mindspore.communication.comm_func import broadcast
        >>> import numpy as np
        >>> # Launch 2 processes.
        >>>
        >>> init()
        >>> data = ms.Tensor(np.arange(8).reshape([2, 4]).astype(np.float32))
        >>> out = broadcast(tensor=data, src=0)
        [[0. 1. 2. 3.]
         [4. 5. 6. 7.]]

    Tutorial Examples:
        - `Distributed Set Communication Primitives - Broadcast
          <https://www.mindspore.cn/docs/en/r2.3.1/api_python/samples/ops/communicate_ops.html#broadcast>`_

    """
    if not isinstance(tensor, (Tensor, Tensor_)):
        raise TypeError("For broadcast, the input tensor must be tensor")
    if not isinstance(src, int):
        raise TypeError("For broadcast, the src must be int")
    _src = get_group_rank_from_world_rank(src, group)
    _op = _get_cache_prim(ops.Broadcast)(_src, group)
    return _op((tensor,))[0]
