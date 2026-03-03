"""Random number generation and seed management."""

import numpy as np
from ._device import device as Device
from ._dtype import uint8

# Global RNG state
_cpu_rng_state = None
_initial_seed = None
_npu_seed = None


def _get_cpu_rng():
    """Get or create the global CPU RNG state."""
    global _cpu_rng_state
    if _cpu_rng_state is None:
        _cpu_rng_state = np.random.RandomState()
    return _cpu_rng_state


def manual_seed(seed: int):
    """Set the seed for generating random numbers on all devices.

    Args:
        seed (int): The desired seed. Must be within the range
            [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff].

    Returns:
        Generator: The default CPU generator.
    """
    global _initial_seed, _cpu_rng_state, _npu_seed

    # Validate seed range
    if seed < -0x8000_0000_0000_0000 or seed > 0xffff_ffff_ffff_ffff:
        raise RuntimeError(f"Seed must be within range [-2^63, 2^64-1], got {seed}")

    # Remap negative seeds to positive
    if seed < 0:
        seed = 0xffff_ffff_ffff_ffff + seed

    _initial_seed = seed
    _cpu_rng_state = np.random.RandomState(seed & 0xffffffff)  # NumPy uses 32-bit seed
    _npu_seed = seed

    # Set NPU seed (stored in npu module, used by ACLNN kernels)
    try:
        from . import npu
        if npu.is_available():
            npu.manual_seed(seed)
    except Exception:
        pass

    return default_generator


def seed():
    """Set the seed for generating random numbers to a random number.

    Returns:
        int: The random seed used.
    """
    import time
    random_seed = int(time.time() * 1000000) & 0xffffffff
    manual_seed(random_seed)
    return random_seed


def initial_seed() -> int:
    """Get the initial seed for generating random numbers.

    Returns:
        int: The initial seed, or 0 if not set.
    """
    return _initial_seed if _initial_seed is not None else 0


def get_rng_state():
    """Get the random number generator state as a tensor.

    Returns:
        Tensor: A ByteTensor containing the RNG state (CPU only).
    """
    from ._creation import tensor

    rng = _get_cpu_rng()
    state = rng.get_state()
    # state is tuple: ('MT19937', ndarray(uint32, 624), int, int, float)
    # Save the uint32 array as raw bytes (uint8 view) for compatibility with PyTorch
    state_bytes = state[1].view(np.uint8)
    pos = np.array([state[2]], dtype=np.int32).view(np.uint8)
    return tensor(np.concatenate([state_bytes, pos]), dtype=uint8)


def set_rng_state(new_state):
    """Set the random number generator state from a tensor.

    Args:
        new_state (Tensor): The desired state (CPU only).
    """
    global _cpu_rng_state

    # Convert to numpy - handle both CPU and NPU tensors
    if hasattr(new_state, 'device') and new_state.device.type != 'cpu':
        new_state = new_state.to('cpu')
    raw = new_state.numpy()
    # Extract uint32 state array and position
    state_array = raw[:624*4].view(np.uint32)
    pos = raw[624*4:624*4+4].view(np.int32)[0]
    _cpu_rng_state = np.random.RandomState()
    _cpu_rng_state.set_state(('MT19937', state_array, int(pos), 0, 0.0))


class Generator:
    """Random number generator for a specific device.

    Args:
        device (str or Device): The device for this generator. Default: 'cpu'
    """

    def __init__(self, device='cpu'):
        self.device = Device(device) if not isinstance(device, Device) else device
        self._seed = None

        if self.device.type == 'cpu':
            self._rng = np.random.RandomState()
        elif self.device.type == 'npu':
            # NPU uses global MindSpore seed
            self._rng = None
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

    def manual_seed(self, seed: int):
        """Set the seed for this generator.

        Args:
            seed (int): The desired seed.

        Returns:
            Generator: self
        """
        # Validate seed range
        if seed < -0x8000_0000_0000_0000 or seed > 0xffff_ffff_ffff_ffff:
            raise RuntimeError(f"Seed must be within range [-2^63, 2^64-1], got {seed}")

        # Remap negative seeds to positive
        if seed < 0:
            seed = 0xffff_ffff_ffff_ffff + seed

        self._seed = seed

        if self.device.type == 'cpu':
            self._rng.seed(seed & 0xffffffff)
        elif self.device.type == 'npu':
            # NPU seed is managed by npu module, used by ACLNN kernels
            try:
                from . import npu
                npu.manual_seed(seed)
            except Exception:
                pass

        return self

    def initial_seed(self) -> int:
        """Get the initial seed for this generator.

        Returns:
            int: The initial seed, or 0 if not set.
        """
        return self._seed if self._seed is not None else 0

    def get_state(self):
        """Get the RNG state for this generator.

        Returns:
            Tensor: The RNG state (CPU only).
        """
        from ._creation import tensor

        if self.device.type == 'cpu':
            state = self._rng.get_state()
            # Save as bytes (uint8 view of uint32 array + position)
            state_bytes = state[1].view(np.uint8)
            pos = np.array([state[2]], dtype=np.int32).view(np.uint8)
            return tensor(np.concatenate([state_bytes, pos]), dtype=uint8)
        else:
            raise NotImplementedError(f"get_state not supported for {self.device.type}")

    def set_state(self, new_state):
        """Set the RNG state for this generator.

        Args:
            new_state (Tensor): The desired state (CPU only).
        """
        if self.device.type == 'cpu':
            if hasattr(new_state, 'device') and new_state.device.type != 'cpu':
                new_state = new_state.to('cpu')
            raw = new_state.numpy()
            # Extract uint32 state array and position
            state_array = raw[:624*4].view(np.uint32)
            pos = raw[624*4:624*4+4].view(np.int32)[0]
            self._rng.set_state(('MT19937', state_array, int(pos), 0, 0.0))
        else:
            raise NotImplementedError(f"set_state not supported for {self.device.type}")


# Default generator (CPU)
default_generator = Generator('cpu')


__all__ = [
    'manual_seed',
    'seed',
    'initial_seed',
    'get_rng_state',
    'set_rng_state',
    'Generator',
    'default_generator',
]
