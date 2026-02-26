"""DataParallel stub for API compatibility.

Single-device passthrough — wraps module and forwards calls.
"""

from ..module import Module


class DataParallel(Module):
    """Stub DataParallel that wraps a module without actual parallelism."""

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        self.module = module
        self.device_ids = device_ids or []
        self.output_device = output_device
        self.dim = dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def _apply(self, fn):
        self.module._apply(fn)
        return self


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    """Functional interface — just runs module on inputs (no parallelism)."""
    if module_kwargs is None:
        module_kwargs = {}
    return module(*inputs, **module_kwargs)
