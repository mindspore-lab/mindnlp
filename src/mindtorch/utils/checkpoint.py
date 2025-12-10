import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from typing import *  # noqa: F403
import enum
from weakref import ReferenceType

import mindtorch
from mindtorch.utils._pytree import tree_map

def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[mindtorch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, mindtorch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(inputs).__name__,
        )

def checkpoint(
    function,
    *args,
    use_reentrant = None,
    context_fn = None,
    determinism_check = None,
    debug = None,
    **kwargs
):
    return function(*args, **kwargs)


def _get_device_module(device="cuda"):
    if device == "meta":
        return mindtorch.device("meta")
    device_module = getattr(mindtorch, device)
    return device_module

class DefaultDeviceType:
    r"""
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """

    _default_device_type = "cuda"

    @staticmethod
    def set_device_type(device: str = "cuda"):
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
        DefaultDeviceType._default_device_type = device

    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """
        return DefaultDeviceType._default_device_type


def _infer_device_type(*args):
    device_types = []

    def add_device_types(arg):
        nonlocal device_types
        if isinstance(arg, mindtorch.Tensor) and arg.device.type != "cpu":
            device_types.append(arg.device.type)
    tree_map(add_device_types, args)

    device_types_set = set(device_types)
    if len(device_types_set) > 1:
        warnings.warn(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. "
            "Device state will only be saved for devices of a single device type, and the remaining "
            "devices will be ignored. Consequently, if any checkpointed functions involve randomness, "
            "this may result in incorrect gradients. (Note that if CUDA devices are among the devices "
            "detected, it will be prioritized; otherwise, the first device encountered will be selected.)"
            f"\nDevice types: {sorted(device_types_set)} first device type: {device_types[0]}"
        )
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif "cuda" in device_types_set:
        return "cuda"
    else:
        return device_types[0]

def get_device_states(*args) -> Tuple[List[int], List[mindtorch.Tensor]]:
    # This will not error out if "arg" is a CPU tensor or a non-tensor type because
    # the conditionals short-circuit.
    fwd_device_ids = []

    def add_device_ids(arg):
        nonlocal fwd_device_ids
        if isinstance(arg, mindtorch.Tensor) and arg.device.type not in {"cpu", "meta"}:
            fwd_device_ids.append(arg.get_device())
    tree_map(add_device_ids, args)

    fwd_device_states = []
    device_module = _get_device_module(_infer_device_type(*args))
    for device_id in fwd_device_ids:
        with device_module.device(device_id):
            fwd_device_states.append(device_module.get_rng_state())

    return fwd_device_ids, fwd_device_states

def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """
    if device_type is None:
        device_type = DefaultDeviceType.get_device_type()
    if device_type == "meta":
        return
    device_module = _get_device_module(device_type)
    for device, state in zip(devices, states):
        with device_module.device(device):
            device_module.set_rng_state(state)