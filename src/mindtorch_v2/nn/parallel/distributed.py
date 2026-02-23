"""DistributedDataParallel implementation for mindtorch_v2.

Bucket-based gradient reduction using tensor.register_hook().
Parameters are grouped into ~25MB buckets. When all grads in a bucket
are ready, the bucket is allreduced (or passed to a custom comm hook).
"""

from contextlib import contextmanager
from ..module import Module


class GradBucket:
    """Matches PyTorch's dist.GradBucket interface for comm hook compatibility."""

    def __init__(self, index, buffer, params, offsets, is_last):
        self._index = index
        self._buffer = buffer
        self._params = params
        self._offsets = offsets
        self._is_last = is_last

    def index(self):
        return self._index

    def buffer(self):
        return self._buffer

    def gradients(self):
        views = []
        for i, param in enumerate(self._params):
            start = self._offsets[i]
            end = self._offsets[i + 1] if i + 1 < len(self._offsets) else self._buffer.numel()
            views.append(self._buffer[start:end].reshape(param.shape))
        return views

    def parameters(self):
        return list(self._params)

    def is_last(self):
        return self._is_last

    def set_buffer(self, tensor):
        self._buffer = tensor


class _Reducer:
    """Bucket-based gradient reducer for DDP.

    When all grads in a bucket are ready, the last arriving hook triggers
    allreduce for the entire bucket and writes averaged grads to param.grad
    for all parameters in the bucket.
    """

    def __init__(self, params, process_group, world_size, bucket_cap_bytes):
        self.process_group = process_group
        self.world_size = world_size
        self.comm_hook = None
        self.comm_hook_state = None
        self._require_backward_grad_sync = True

        # Only keep params that require grad
        self.params = list(params)
        self.grad_params = [(i, p) for i, p in enumerate(self.params) if p.requires_grad]

        # Build buckets (reverse parameter order to match backward order)
        self.buckets = []  # list of [(param_idx, param), ...]
        self._param_to_bucket = {}  # param_idx -> bucket_idx
        self._build_buckets(bucket_cap_bytes)

        # Per-iteration state
        self._bucket_pending = []
        self._bucket_grads = []
        self._reset_state()

    def _build_buckets(self, bucket_cap_bytes):
        cur_bucket = []
        cur_size = 0

        for idx, param in reversed(self.grad_params):
            param_bytes = param.numel() * 4  # assume 4 bytes
            if cur_size + param_bytes > bucket_cap_bytes and cur_bucket:
                self.buckets.append(list(cur_bucket))
                cur_bucket = []
                cur_size = 0
            cur_bucket.append((idx, param))
            cur_size += param_bytes

        if cur_bucket:
            self.buckets.append(list(cur_bucket))

        for bucket_idx, bucket in enumerate(self.buckets):
            for param_idx, _ in bucket:
                self._param_to_bucket[param_idx] = bucket_idx

    def _reset_state(self):
        self._bucket_pending = [len(b) for b in self.buckets]
        self._bucket_grads = [{} for _ in self.buckets]

    def register_hooks(self):
        for idx, param in self.grad_params:
            if idx in self._param_to_bucket:
                param.register_hook(self._make_hook(idx))

    def _make_hook(self, param_idx):
        def hook(grad):
            if not self._require_backward_grad_sync:
                return grad
            bucket_idx = self._param_to_bucket[param_idx]
            self._bucket_grads[bucket_idx][param_idx] = grad
            self._bucket_pending[bucket_idx] -= 1
            if self._bucket_pending[bucket_idx] == 0:
                self._reduce_bucket(bucket_idx)
                # Return the averaged grad for this param
                return self._bucket_grads[bucket_idx][param_idx]
            # Not the last param in bucket — allreduce hasn't happened yet.
            # We return grad as-is; it will be overwritten after the bucket
            # is reduced (via param.grad assignment in _reduce_bucket).
            return grad
        return hook

    def _reduce_bucket(self, bucket_idx):
        from ... import distributed as dist
        from ..._functional import mul
        from ..._autograd.grad_mode import no_grad

        bucket = self.buckets[bucket_idx]
        grads = self._bucket_grads[bucket_idx]

        with no_grad():
            if self.comm_hook is not None:
                self._run_comm_hook(bucket_idx, bucket, grads)
            else:
                self._run_default_allreduce(bucket, grads, dist)

            # Write averaged grads back to all params in bucket.
            # For the last-arriving param, the hook returns the updated grad.
            # For earlier params, their hook already returned the original grad
            # (which the engine assigned to param.grad), so we overwrite here.
            for pi, param in bucket:
                param.grad = grads[pi]

    def _run_default_allreduce(self, bucket, grads, dist):
        from ..._functional import mul
        for pi, _ in bucket:
            g = grads[pi]
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=self.process_group)
            grads[pi] = mul(g, 1.0 / self.world_size)

    def _run_comm_hook(self, bucket_idx, bucket, grads):
        # Call comm hook per-parameter (avoids needing cat on NPU)
        for i, (pi, param) in enumerate(bucket):
            g = grads[pi]
            grad_bucket = GradBucket(
                index=bucket_idx,
                buffer=g,
                params=[param],
                offsets=[0],
                is_last=(bucket_idx == len(self.buckets) - 1 and i == len(bucket) - 1),
            )
            future = self.comm_hook(self.comm_hook_state, grad_bucket)
            grads[pi] = future.wait()

    def _find_unused_params(self, outputs):
        """Mark unused parameters' buckets as ready with zero gradients."""
        from ..._functional import zeros_like
        from ..._autograd.grad_mode import no_grad

        # Flatten outputs to tensors with grad_fn
        if isinstance(outputs, dict):
            tensors = [v for v in outputs.values() if hasattr(v, 'grad_fn') and v.grad_fn is not None]
        elif isinstance(outputs, (tuple, list)):
            tensors = [t for t in outputs if hasattr(t, 'grad_fn') and t.grad_fn is not None]
        elif hasattr(outputs, 'grad_fn') and outputs.grad_fn is not None:
            tensors = [outputs]
        else:
            return

        # Traverse autograd graph, collect ids of all reachable leaf tensors
        used = set()
        visited = set()
        stack = [t.grad_fn for t in tensors]
        while stack:
            node = stack.pop()
            if id(node) in visited:
                continue
            visited.add(id(node))
            for inp in node.inputs:
                if inp.grad_fn is None:
                    used.add(id(inp))
                else:
                    stack.append(inp.grad_fn)

        # For each unused grad param, simulate hook firing with zero grad
        with no_grad():
            for idx, param in self.grad_params:
                if id(param) not in used:
                    bucket_idx = self._param_to_bucket.get(idx)
                    if bucket_idx is not None:
                        self._bucket_grads[bucket_idx][idx] = zeros_like(param)
                        self._bucket_pending[bucket_idx] -= 1
                        if self._bucket_pending[bucket_idx] == 0:
                            self._reduce_bucket(bucket_idx)

    def prepare_for_backward(self):
        self._reset_state()

    def register_comm_hook(self, state, hook):
        self.comm_hook = hook
        self.comm_hook_state = state


class DistributedDataParallel(Module):

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    ):
        super().__init__()
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        self.dim = dim
        self.bucket_cap_mb = bucket_cap_mb

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

        # Build reducer with bucket-based grad sync
        bucket_cap_bytes = bucket_cap_mb * 1024 * 1024
        self.reducer = _Reducer(
            list(module.parameters()),
            process_group,
            self.world_size,
            bucket_cap_bytes,
        )
        self.reducer.register_hooks()

    def _sync_params_and_buffers(self):
        from ... import distributed as dist
        tensors = list(self.module.parameters()) + list(self.module.buffers())
        if tensors:
            dist._broadcast_coalesced(tensors, src=0, group=self.process_group)

    def forward(self, *args, **kwargs):
        if self.broadcast_buffers and self.world_size > 1:
            from ... import distributed as dist
            buffers = list(self.module.buffers())
            if buffers:
                dist._broadcast_coalesced(buffers, src=0, group=self.process_group)

        self.reducer._require_backward_grad_sync = self._require_backward_grad_sync
        self.reducer.prepare_for_backward()
        output = self.module(*args, **kwargs)
        if self.find_unused_parameters and self._require_backward_grad_sync:
            self.reducer._find_unused_params(output)
        return output

    @contextmanager
    def no_sync(self):
        old = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = old

    def register_comm_hook(self, state, hook):
        """Register a communication hook for custom gradient reduction.

        Args:
            state: Passed to hook. Used to maintain state across calls.
            hook: Callable with signature hook(state, bucket: GradBucket) -> Future[Tensor]
        """
        self.reducer.register_comm_hook(state, hook)

    # --- Delegation to wrapped module ---

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse=True):
        return self.module.buffers(recurse=recurse)

    def named_buffers(self, prefix='', recurse=True):
        return self.module.named_buffers(prefix=prefix, recurse=recurse)

    def children(self):
        return self.module.children()

    def named_children(self):
        return self.module.named_children()

    def modules(self):
        return self.module.modules()

    def named_modules(self, memo=None, prefix=''):
        return self.module.named_modules(memo=memo, prefix=prefix)

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
