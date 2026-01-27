"""Stub for torch.cuda - GPU/CUDA operations.

Tier 2 stub: called but returns no-op values indicating no CUDA support.
"""


def is_available():
    """Check if CUDA is available."""
    return False


def device_count():
    """Return number of CUDA devices."""
    return 0


def current_device():
    """Return current CUDA device index."""
    return 0


def get_device_name(device=None):
    """Return device name."""
    return "cpu"


def get_device_capability(device=None):
    """Return device compute capability."""
    return (0, 0)


def synchronize(device=None):
    """Synchronize CUDA device."""
    pass


def set_device(device):
    """Set current CUDA device."""
    pass


def memory_allocated(device=None):
    """Return memory allocated on device."""
    return 0


def max_memory_allocated(device=None):
    """Return max memory allocated on device."""
    return 0


def memory_reserved(device=None):
    """Return memory reserved on device."""
    return 0


def max_memory_reserved(device=None):
    """Return max memory reserved on device."""
    return 0


def empty_cache():
    """Empty CUDA cache."""
    pass


def reset_peak_memory_stats(device=None):
    """Reset peak memory stats."""
    pass


def reset_max_memory_allocated(device=None):
    """Reset max memory allocated."""
    pass


def reset_max_memory_cached(device=None):
    """Reset max memory cached."""
    pass


class Stream:
    """CUDA stream stub."""

    def __init__(self, device=None, priority=0):
        self.device = device

    def synchronize(self):
        pass

    def wait_event(self, event):
        pass

    def wait_stream(self, stream):
        pass

    def record_event(self, event=None):
        return Event()

    def query(self):
        return True


class Event:
    """CUDA event stub."""

    def __init__(self, enable_timing=False):
        pass

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, end_event):
        return 0.0


def stream(stream=None):
    """Context manager for CUDA stream."""
    from contextlib import nullcontext
    return nullcontext()


def current_stream(device=None):
    """Return current CUDA stream."""
    return Stream()


def default_stream(device=None):
    """Return default CUDA stream."""
    return Stream()


class device:
    """Context manager for CUDA device."""

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# AMP (Automatic Mixed Precision) related
class amp:
    """CUDA AMP namespace."""

    @staticmethod
    def autocast(enabled=True, dtype=None):
        from contextlib import nullcontext
        return nullcontext()

    class GradScaler:
        def __init__(self, enabled=True):
            self.enabled = enabled

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

        def get_scale(self):
            return 1.0

        def is_enabled(self):
            return self.enabled


# NCCL related
class nccl:
    """NCCL stub."""

    @staticmethod
    def is_available(tensors=None):
        return False

    @staticmethod
    def version():
        return (0, 0, 0)


# cuDNN related
class cudnn:
    """cuDNN stub - see backends.py for full implementation."""
    enabled = False
    benchmark = False
    deterministic = True
    allow_tf32 = False

    @staticmethod
    def is_available():
        return False

    @staticmethod
    def version():
        return None


# Additional utilities
def get_rng_state(device=None):
    """Get RNG state."""
    return b''


def set_rng_state(new_state, device=None):
    """Set RNG state."""
    pass


def manual_seed(seed):
    """Set CUDA random seed."""
    pass


def manual_seed_all(seed):
    """Set CUDA random seed for all devices."""
    pass


def seed():
    """Seed CUDA RNG."""
    pass


def seed_all():
    """Seed CUDA RNG for all devices."""
    pass


def initial_seed():
    """Return initial CUDA seed."""
    return 0


# BFloat16 support
class BFloat16Storage:
    pass


def is_bf16_supported():
    """Check if BFloat16 is supported."""
    return False


# Flash attention
def can_use_flash_attention(*args, **kwargs):
    """Check if flash attention can be used."""
    return False


def can_use_efficient_attention(*args, **kwargs):
    """Check if efficient attention can be used."""
    return False


# Device properties
def get_device_properties(device=None):
    """Get device properties."""

    class DeviceProperties:
        name = "cpu"
        major = 0
        minor = 0
        total_memory = 0
        multi_processor_count = 0

    return DeviceProperties()
