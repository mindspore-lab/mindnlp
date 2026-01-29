"""Dummy parallel module implementations for compatibility.

These are placeholder classes that don't actually implement parallelism
but allow code that expects these classes to exist to work.
"""

from ..module import Module


class DataParallel(Module):
    """Dummy DataParallel that just wraps a module.

    This doesn't implement actual data parallelism but provides
    compatibility for code that checks isinstance(model, DataParallel).
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


class DistributedDataParallel(Module):
    """Dummy DistributedDataParallel that just wraps a module.

    This doesn't implement actual distributed data parallelism but provides
    compatibility for code that checks isinstance(model, DistributedDataParallel).
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0,
                 broadcast_buffers=True, process_group=None, bucket_cap_mb=25,
                 find_unused_parameters=False, check_reduction=False,
                 gradient_as_bucket_view=False, static_graph=False):
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
