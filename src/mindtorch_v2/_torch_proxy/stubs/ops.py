"""Stub for torch.ops module."""


class _Ops:
    """Stub for torch.ops - provides no-op implementations."""

    def __getattr__(self, name):
        """Return a namespace for any attribute access."""
        return _OpsNamespace(name)

    def load_library(self, path):
        """Load a custom operator library - raise error to indicate not available."""
        raise OSError(f"Cannot load native library: {path} (mindtorch_v2 does not support native extensions)")


class _OpsNamespace:
    """Namespace stub for torch.ops.xxx."""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, name):
        """Return a callable stub for any operation."""
        return _OpStub(f"{self._name}.{name}")


class _OpStub:
    """Stub for individual operators."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        """Handle operator calls - return safe defaults for known ops."""
        # Handle torchvision CUDA version check
        if '_cuda_version' in self._name:
            return -1  # Indicate no CUDA
        if '_is_compiled' in self._name:
            return False
        # Default: raise error for unknown ops
        raise NotImplementedError(
            f"torch.ops.{self._name} not available in mindtorch_v2"
        )

    def __getattr__(self, name):
        """Return another stub for chained access."""
        return _OpStub(f"{self._name}.{name}")


# Module-level instance
_ops_instance = _Ops()


def load_library(path):
    """Load a custom operator library - raise error to indicate not available."""
    raise OSError(f"Cannot load native library: {path} (mindtorch_v2 does not support native extensions)")


def __getattr__(name):
    """Module-level getattr for torch.ops.xxx access."""
    if name == 'load_library':
        return load_library
    return getattr(_ops_instance, name)
