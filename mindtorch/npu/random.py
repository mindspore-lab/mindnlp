from typing import Union

import mindtorch


def get_rng_state(device: Union[int, str, mindtorch.device] = "npu") -> mindtorch.Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (msadapter.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``msadapter.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    
    if isinstance(device, str):
        device = mindtorch.device(device)
    elif isinstance(device, int):
        device = mindtorch.device("npu", device)
    idx = device.index
    default_generator = mindtorch.npu.default_generators[idx]
    return default_generator.get_state()
