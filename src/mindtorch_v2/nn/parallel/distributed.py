"""DistributedDataParallel implementation for mindtorch_v2.

Pure Python implementation using tensor.register_hook() for gradient synchronization.
Each parameter's gradient is allreduced inline during backward via hooks.
"""

from contextlib import contextmanager
from ..module import Module


class DistributedDataParallel(Module):

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    ):
        super().__init__()
        self.module = module
        self.broadcast_buffers = broadcast_buffers

        from ... import distributed as dist
        if process_group is None:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Default process group has not been initialized, "
                    "please make sure to call init_process_group."
                )
            process_group = dist.group.WORLD
        self.process_group = process_group
        self.world_size = dist.get_world_size(self.process_group)

        self._require_backward_grad_sync = True

        # Broadcast params and buffers from rank 0
        self._sync_params_and_buffers()

        # Register allreduce hooks on parameters
        for param in module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook())

    def _sync_params_and_buffers(self):
        from ... import distributed as dist
        tensors = list(self.module.parameters()) + list(self.module.buffers())
        if tensors:
            dist._broadcast_coalesced(tensors, src=0, group=self.process_group)

    def _make_hook(self):
        def hook(grad):
            if not self._require_backward_grad_sync:
                return grad
            from ... import distributed as dist
            from ..._functional import mul
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.process_group)
            return mul(grad, 1.0 / self.world_size)
        return hook

    def forward(self, *args, **kwargs):
        if self.broadcast_buffers and self.world_size > 1:
            from ... import distributed as dist
            buffers = list(self.module.buffers())
            if buffers:
                dist._broadcast_coalesced(buffers, src=0, group=self.process_group)
        return self.module(*args, **kwargs)

    @contextmanager
    def no_sync(self):
        old = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = old

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

    def train(self, mode=True):
        super().train(mode)
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)
