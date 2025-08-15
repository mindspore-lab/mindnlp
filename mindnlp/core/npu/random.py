from typing import Union

from mindnlp import core


def get_rng_state(device: Union[int, str, core.device] = "npu") -> core.Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (msadapter.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``msadapter.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    
    if isinstance(device, str):
        device = core.device(device)
    elif isinstance(device, int):
        device = core.device("npu", device)
    idx = device.index
    default_generator = core.npu.default_generators[idx]
    return default_generator.get_state()
