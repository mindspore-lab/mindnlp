"""big modeling"""
from contextlib import contextmanager
from mindspore._c_expression import Tensor as Tensor_ # pylint: disable=no-name-in-module
from mindnlp.utils.testing_utils import parse_flag_from_env
from mindnlp.core import nn

@contextmanager
def init_empty_weights(include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the meta device, therefore creating an
    empty model. Useful when just initializing the model would blow the available RAM.

    Args:
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_empty_weights

    # Initialize a model with 100 billions parameters in no time and without using any RAM.
    with init_empty_weights():
        tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
    ```

    <Tip warning={true}>

    Any model created under this context manager has no weights. As such you can't do something like
    `model.to(some_device)` with it. To load weights inside your empty model, see [`load_checkpoint_and_dispatch`].
    Make sure to overwrite the default device_map param for [`load_checkpoint_and_dispatch`], otherwise dispatch is not
    called.

    </Tip>
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)
    with init_on_empty(include_buffers=include_buffers) as f:
        yield f


@contextmanager
def init_on_empty(include_buffers: bool = None):
    """
    A context manager under which models are initialized with all parameters on the specified device.

    Args:
        device (`torch.device`):
            Device to initialize all parameters on.
        include_buffers (`bool`, *optional*):
            Whether or not to also put all buffers on the meta device while initializing.

    Example:

    ```python
    import torch.nn as nn
    from accelerate import init_on_device

    with init_on_device(device=torch.device("cuda")):
        tst = nn.Linear(100, 100)  # on `cuda` device
    ```
    """
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name].assign_value(Tensor_(shape=module._parameters[name].shape, dtype=module._parameters[name].dtype))
            module._parameters[name].meta = True

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name].assign_value(Tensor_(shape=module._parameters[name].shape, dtype=module._buffers[name].dtype))
            module._buffers[name].meta = True

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
