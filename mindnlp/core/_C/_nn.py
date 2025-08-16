from mindnlp import core

def _parse_to(*args, **kwargs):
    """
    Mimic core._C._nn._parse_to functionality in Python.
    
    Args:
        tensor (core.Tensor): The tensor to parse.
        *args: Positional arguments for `to`.
        **kwargs: Keyword arguments for `to`.

    Returns:
        core.Tensor: The tensor with the desired properties.
    """
    if len(args) == 1:
        # Handle `device` or `dtype`
        if isinstance(args[0], core.dtype):  # dtype only
            dtype = args[0]
            device = None
        elif isinstance(args[0], core.device):  # device only
            device = args[0]
            dtype = None
        elif isinstance(args[0], (str, int)):
            device = core.device(args[0])
            dtype = None
        else:
            raise TypeError(f"Expected core.dtype or core.device, but got {type(args[0])}")
    elif len(args) == 2:
        # Handle `device` and `dtype`
        dtype = args[1]
        device = args[0]
    else:
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)

    non_blocking = kwargs.get("non_blocking", False)
    memory_format = kwargs.get("memory_format", None)

    return device, dtype, non_blocking, memory_format
