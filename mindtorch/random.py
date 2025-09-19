# mypy: allow-untyped-defs
import contextlib
import warnings
from typing import Generator

import mindspore
import mindtorch
from ._C import default_generator
# from mindspore import default_generator, set_seed


def get_rng_state():
    """
    Get the state of the default generator.

    Returns:
        Tensor, generator state.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import get_rng_state
        >>> state = get_rng_state()
    """
    return default_generator.get_state()


def set_rng_state(state):  # pylint: disable=redefined-outer-name
    """
    Set the state of the default generator.

    Args:
        state (Tensor): the target state

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import set_rng_state, get_rng_state
        >>> state = get_rng_state()
        >>> set_rng_state(state)
    """
    default_generator.set_state(state)

def manual_seed(seed):
    r"""Sets the seed for generating random numbers on all devices. Returns a
    `mindtorch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """
    # mindspore.set_seed(seed + 1)
    seed = int(seed)
    # set_seed(seed)
    return default_generator.manual_seed(seed)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64 bit number used to seed the RNG.
    """
    seed = default_generator.seed()

    return seed


def initial_seed() -> int:
    r"""Returns the initial seed for generating random numbers as a
    Python `long`.

    .. note:: The returned seed is for the default generator on CPU only.
    """
    return default_generator.initial_seed()


_fork_rng_warned_already = False


@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type="cuda",
) -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is `cuda`. As for custom device,
            see details in [Note: support the custom device with privateuse1]
    """

    if device_type == "meta":
        yield
        return

    device_type = mindtorch.device(device_type).type
    device_mod = getattr(torch, device_type, None)
    if device_mod is None:
        raise RuntimeError(
            f"torch has no module of `{device_type}`, you should register "
            + "a module by `mindtorch._register_device_module`."
        )
    global _fork_rng_warned_already

    # Internal arguments:
    #   _caller: the function which called fork_rng, which the user used
    #   _devices_kw: the devices keyword of _caller

    if not enabled:
        yield
        return

    if devices is None:
        num_devices = device_mod.device_count()
        if num_devices > 1 and not _fork_rng_warned_already:
            message = (
                f"{device_type.upper()} reports that you have {num_devices} available devices, and "
                f"you have used {_caller} without explicitly specifying which devices are being used. "
                f"For safety, we initialize *every* {device_type.upper()} device by default, which can "
                f"be quite slow if you have a lot of {device_type.upper()}s. If you know that you are only"
                f" making use of a few {device_type.upper()} devices, set the environment variable "
                f"{device_type.upper()}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} "
                "with the set of devices you are actually using. For example, if you are using CPU only, "
                "set device.upper()_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, "
                f"set {device_type.upper()}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices "
                f"and suppress this warning, set the '{_devices_kw}' keyword argument to "
                f"`range(mindtorch.{device_type}.device_count())`."
            )
            warnings.warn(message)
            _fork_rng_warned_already = True
        devices = list(range(num_devices))
    else:
        # Protect against user passing us a generator; we need to traverse this
        # multiple times but a generator will be exhausted upon first traversal
        devices = list(devices)

    cpu_rng_state = mindtorch.get_rng_state()
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]

    try:
        yield
    finally:
        mindtorch.set_rng_state(cpu_rng_state)
        for device, device_rng_state in zip(devices, device_rng_states):
            device_mod.set_rng_state(device_rng_state, device)
