"""Random number generation and seed management."""

import contextlib
import numpy as np
from ._device import device as Device
from ._dtype import uint8


def _get_cpu_rng():
    """Get the CPU RNG state from the default generator."""
    return default_generator._rng


def manual_seed(seed: int):
    """Set the seed for generating random numbers on all devices.

    Args:
        seed (int): The desired seed. Must be within the range
            [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff].

    Returns:
        Generator: The default CPU generator.
    """
    # Seed the CPU default generator
    default_generator.manual_seed(seed)

    # Propagate to all NPU devices
    try:
        from . import npu
        if npu.is_available():
            npu.manual_seed_all(seed)
    except Exception:
        pass

    return default_generator


def seed():
    """Set the seed for generating random numbers to a random number.

    Returns:
        int: The random seed used.
    """
    s = default_generator.seed()
    # Propagate to NPU
    try:
        from . import npu
        if npu.is_available():
            npu.manual_seed_all(s)
    except Exception:
        pass
    return s


def initial_seed() -> int:
    """Get the initial seed for the default CPU generator.

    Returns:
        int: The initial seed.
    """
    return default_generator.initial_seed()


def get_rng_state():
    """Get the random number generator state as a tensor.

    Returns:
        Tensor: A ByteTensor containing the RNG state (CPU only).
    """
    return default_generator.get_state()


def set_rng_state(new_state):
    """Set the random number generator state from a tensor.

    Args:
        new_state (Tensor): The desired state (CPU only).
    """
    default_generator.set_state(new_state)


class Generator:
    """Random number generator for a specific device.

    Args:
        device (str or Device): The device for this generator. Default: 'cpu'
    """

    # PyTorch's default seed constant
    _DEFAULT_SEED = 67280421310721

    def __init__(self, device='cpu'):
        self.device = Device(device) if not isinstance(device, Device) else device
        self._seed = self._DEFAULT_SEED

        if self.device.type == 'cpu':
            self._rng = np.random.RandomState(self._seed & 0xffffffff)
        elif self.device.type == 'npu':
            self._rng = None
            self._offset = 0
        else:
            raise ValueError(f"Unsupported device type: {self.device.type}")

    def manual_seed(self, seed: int):
        """Set the seed for this generator."""
        if seed < -0x8000_0000_0000_0000 or seed > 0xffff_ffff_ffff_ffff:
            raise RuntimeError(
                f"Seed must be within range [-2^63, 2^64-1], got {seed}"
            )
        if seed < 0:
            seed = 0xffff_ffff_ffff_ffff + seed

        self._seed = seed

        if self.device.type == 'cpu':
            self._rng = np.random.RandomState(seed & 0xffffffff)
        elif self.device.type == 'npu':
            self._offset = 0
        return self

    def seed(self):
        """Set the seed to a random value and return it."""
        import time
        s = int(time.time() * 1000000) & 0xffff_ffff_ffff_ffff
        self.manual_seed(s)
        return s

    def initial_seed(self) -> int:
        """Get the current seed for this generator."""
        return self._seed

    def get_state(self):
        """Get the RNG state as a ByteTensor."""
        from ._creation import tensor

        if self.device.type == 'cpu':
            state = self._rng.get_state()
            state_bytes = state[1].view(np.uint8)
            pos = np.array([state[2]], dtype=np.int32).view(np.uint8)
            return tensor(np.concatenate([state_bytes, pos]), dtype=uint8)
        elif self.device.type == 'npu':
            # NPU state: seed (uint64) + offset (int64) = 16 bytes
            buf = np.zeros(16, dtype=np.uint8)
            buf[:8] = np.array([self._seed], dtype=np.uint64).view(np.uint8)
            buf[8:] = np.array([self._offset], dtype=np.int64).view(np.uint8)
            return tensor(buf, dtype=uint8)

    def set_state(self, new_state):
        """Set the RNG state from a ByteTensor."""
        if hasattr(new_state, 'device') and new_state.device.type != 'cpu':
            new_state = new_state.to('cpu')
        raw = new_state.numpy()

        if self.device.type == 'cpu':
            state_array = raw[:624 * 4].view(np.uint32)
            pos = raw[624 * 4:624 * 4 + 4].view(np.int32)[0]
            self._rng = np.random.RandomState()
            self._rng.set_state(('MT19937', state_array, int(pos), 0, 0.0))
        elif self.device.type == 'npu':
            self._seed = int(raw[:8].view(np.uint64)[0])
            self._offset = int(raw[8:16].view(np.int64)[0])

    def philox_engine_inputs(self, increment=10):
        """Get (seed, offset) for NPU ACLNN kernels and advance offset.

        This matches torch_npu's NPUGeneratorImpl::philox_engine_inputs().
        The increment represents Philox rounds of separation between ops.

        Args:
            increment: Number of Philox rounds to advance. Default: 10.

        Returns:
            tuple: (seed, offset) as integers.
        """
        if self.device.type != 'npu':
            raise RuntimeError("philox_engine_inputs only for NPU generators")
        seed = self._seed
        offset = self._offset
        self._offset += increment
        return seed, offset


# Default generator (CPU)
default_generator = Generator('cpu')


def bernoulli(input, *, generator=None):
    """Sample Bernoulli distribution given probabilities tensor.

    Args:
        input (Tensor): Probability values in [0, 1].
        generator (Generator, optional): RNG generator.

    Returns:
        Tensor: Binary tensor of same shape as input.
    """
    from ._creation import tensor
    from ._dtype import float32

    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    if hasattr(input, '_numpy_view'):
        probs = input._numpy_view().copy()
    else:
        probs = np.array(input, dtype=np.float32)
    uniform = rng.uniform(0.0, 1.0, size=probs.shape)
    out = (uniform < probs).astype(probs.dtype)
    return tensor(out, dtype=input.dtype if hasattr(input, 'dtype') else float32)


def multinomial(input, num_samples, replacement=False, *, generator=None):
    """Sample indices from a multinomial distribution.

    Args:
        input (Tensor): 1D or 2D tensor of weights (unnormalized probabilities).
        num_samples (int): Number of samples to draw per row.
        replacement (bool): Whether to sample with replacement. Default: False.
        generator (Generator, optional): RNG generator.

    Returns:
        Tensor: LongTensor of sampled indices.
    """
    from ._creation import tensor
    from ._dtype import int64

    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    if hasattr(input, '_numpy_view'):
        weights = input._numpy_view().copy().astype(np.float64)
    else:
        weights = np.array(input, dtype=np.float64)

    if weights.ndim == 1:
        total = weights.sum()
        if total == 0:
            probs = np.ones_like(weights) / len(weights)
        else:
            probs = weights / total
        indices = rng.choice(len(probs), size=num_samples, replace=replacement, p=probs)
        return tensor(indices.astype(np.int64), dtype=int64)
    elif weights.ndim == 2:
        rows = weights.shape[0]
        out = np.zeros((rows, num_samples), dtype=np.int64)
        for i in range(rows):
            row = weights[i]
            total = row.sum()
            if total == 0:
                probs = np.ones_like(row) / len(row)
            else:
                probs = row / total
            out[i] = rng.choice(len(probs), size=num_samples, replace=replacement, p=probs)
        return tensor(out, dtype=int64)
    else:
        raise ValueError("multinomial expects 1D or 2D input")


def poisson(lam, generator=None):
    """Sample from a Poisson distribution with rate parameter lam."""
    from ._creation import tensor
    from ._dtype import float32

    rng = generator._rng if (generator is not None and hasattr(generator, '_rng') and generator._rng is not None) else _get_cpu_rng()
    if hasattr(lam, '_numpy_view'):
        rates = lam._numpy_view().copy().astype(np.float64)
    elif hasattr(lam, 'numpy'):
        rates = lam.numpy().astype(np.float64)
    else:
        rates = np.array(lam, dtype=np.float64)
    out = rng.poisson(rates).astype(np.float32)
    out_dtype = lam.dtype if hasattr(lam, 'dtype') else float32
    return tensor(out, dtype=out_dtype)


@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller='fork_rng', _devices_kw='devices'):
    """Fork the RNG state so it is restored upon exiting the context.

    Args:
        devices: Ignored (for API compatibility with PyTorch CUDA fork_rng).
        enabled (bool): If False, the context manager is a no-op. Default: True.
    """
    if not enabled:
        yield
        return
    # Save CPU state
    cpu_state = default_generator.get_state()
    try:
        yield
    finally:
        # Restore CPU state
        default_generator.set_state(cpu_state)


__all__ = [
    'manual_seed',
    'seed',
    'initial_seed',
    'get_rng_state',
    'set_rng_state',
    'Generator',
    'default_generator',
    'bernoulli',
    'multinomial',
    'poisson',
    'fork_rng',
]
