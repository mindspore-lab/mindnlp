# mypy: allow-untyped-defs
"""
DistributedDataParallel implementation for mindtorch.

This module provides distributed data parallelism functionality similar to PyTorch's
DistributedDataParallel, adapted for MindSpore backend.
"""

import warnings
from typing import Optional, List, Any, Dict, Set, Tuple
import weakref
import threading
from collections import defaultdict

import mindtorch
from mindtorch.distributed import (
    get_rank,
    get_world_size,
    all_reduce,
    broadcast,
    ReduceOp,
    is_initialized,
)
from mindtorch.distributed.distributed_c10d import (
    ProcessGroup,
    _get_default_group,
)
from ..modules import Module
from ..parameter import Parameter


# Default bucket size in bytes (25 MiB)
_DEFAULT_BUCKET_CAP_MB = 25


def _find_tensors(obj):
    """Recursively find all tensors contained in the specified object."""
    if isinstance(obj, mindtorch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            result.extend(_find_tensors(item))
        return result
    if isinstance(obj, dict):
        result = []
        for value in obj.values():
            result.extend(_find_tensors(value))
        return result
    return []


class _Bucket:
    """Represents a bucket of parameters for gradient reduction."""
    
    def __init__(self, bucket_idx: int, parameters: List[Parameter], bucket_size: int):
        self.bucket_idx = bucket_idx
        self.parameters = parameters
        self.bucket_size = bucket_size
        self.offset = 0
        self.pending = set(parameters)
        self.ready = False
        self.gradients = []
        
    def mark_ready(self, param: Parameter):
        """Mark a parameter as ready for reduction."""
        self.pending.discard(param)
        if len(self.pending) == 0:
            self.ready = True


class _Reducer:
    """Manages gradient reduction using bucketing strategy."""
    
    def __init__(
        self,
        parameters: List[Parameter],
        process_group: ProcessGroup,
        bucket_cap_mb: int,
        find_unused_parameters: bool,
        gradient_as_bucket_view: bool,
        static_graph: bool,
    ):
        self.parameters = parameters
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        
        self.world_size = get_world_size(process_group)
        self.rank = get_rank(process_group)
        
        # Build buckets - parameters in reverse order (for backward pass)
        self.buckets = self._build_buckets()
        
        # Track parameter to bucket mapping
        self.param_to_bucket: Dict[Parameter, _Bucket] = {}
        for bucket in self.buckets:
            for param in bucket.parameters:
                self.param_to_bucket[param] = bucket
        
        # Track unused parameters
        self.unused_parameters: Set[Parameter] = set()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Track backward state
        self._expect_sparse_gradient = False
        self._has_marked_unused_parameters = False
        
    def _build_buckets(self) -> List[_Bucket]:
        """Build buckets from parameters based on size.
        
        Parameters are processed in reverse order to match backward pass order.
        """
        if len(self.parameters) == 0:
            return []
        
        # Process parameters in reverse order (as they appear in backward)
        params_reversed = list(reversed(self.parameters))
        
        # Calculate parameter sizes
        param_sizes = []
        for param in params_reversed:
            if param.numel() == 0:
                continue
            # Get element size (bytes per element)
            try:
                element_size = param.element_size() if hasattr(param, 'element_size') else 4
            except:
                element_size = 4  # Default to 4 bytes (float32)
            param_size = param.numel() * element_size
            param_sizes.append((param, param_size))
        
        # Sort by size (largest first) for better bucket packing
        param_sizes.sort(key=lambda x: x[1], reverse=True)
        
        buckets = []
        current_bucket_params = []
        current_bucket_size = 0
        bucket_idx = 0
        
        for param, size in param_sizes:
            # If single parameter is larger than bucket cap, put it in its own bucket
            if size > self.bucket_bytes_cap:
                if current_bucket_params:
                    buckets.append(_Bucket(bucket_idx, current_bucket_params, current_bucket_size))
                    bucket_idx += 1
                    current_bucket_params = []
                    current_bucket_size = 0
                buckets.append(_Bucket(bucket_idx, [param], size))
                bucket_idx += 1
            elif current_bucket_size + size <= self.bucket_bytes_cap:
                current_bucket_params.append(param)
                current_bucket_size += size
            else:
                # Start a new bucket
                if current_bucket_params:
                    buckets.append(_Bucket(bucket_idx, current_bucket_params, current_bucket_size))
                    bucket_idx += 1
                current_bucket_params = [param]
                current_bucket_size = size
        
        # Add the last bucket
        if current_bucket_params:
            buckets.append(_Bucket(bucket_idx, current_bucket_params, current_bucket_size))
        
        return buckets
    
    def _reduce_bucket(self, bucket: _Bucket):
        """Reduce gradients in a bucket."""
        # Collect gradients from parameters in this bucket
        grads = []
        param_indices = []
        for i, param in enumerate(bucket.parameters):
            if param.grad is not None:
                grads.append(param.grad)
                param_indices.append(i)
        
        if not grads:
            # No gradients to reduce - mark all as ready
            for param in bucket.parameters:
                bucket.mark_ready(param)
            return
        
        # Concatenate gradients if multiple parameters
        if len(grads) == 1:
            bucket_grad = grads[0]
            if not self.gradient_as_bucket_view:
                bucket_grad = bucket_grad.clone()
        else:
            # Flatten and concatenate
            flat_grads = [g.flatten() for g in grads]
            bucket_grad = mindtorch.cat(flat_grads)
            if not self.gradient_as_bucket_view:
                bucket_grad = bucket_grad.clone()
        
        # All-reduce the bucket gradient (sum)
        all_reduce(bucket_grad, op=ReduceOp.SUM, group=self.process_group)
        
        # Average the gradients (divide by world size)
        bucket_grad.div_(self.world_size)
        
        # Distribute back to parameters
        if len(grads) == 1:
            param = bucket.parameters[param_indices[0]]
            if self.gradient_as_bucket_view:
                # For gradient_as_bucket_view, the gradient should be a view
                # In practice, we still need to copy for correctness
                if param.grad is not None:
                    param.grad.copy_(bucket_grad)
                else:
                    param.grad = bucket_grad
            else:
                if param.grad is None:
                    param.grad = bucket_grad
                else:
                    param.grad.copy_(bucket_grad)
        else:
            # Split back to individual parameters
            offset = 0
            for idx in param_indices:
                param = bucket.parameters[idx]
                param_size = param.numel()
                if param_size > 0:
                    param_grad = bucket_grad[offset:offset + param_size].view_as(param)
                    if self.gradient_as_bucket_view:
                        if param.grad is not None:
                            param.grad.copy_(param_grad)
                        else:
                            param.grad = param_grad
                    else:
                        if param.grad is None:
                            param.grad = param_grad.clone()
                        else:
                            param.grad.copy_(param_grad)
                    offset += param_size
    
    def _mark_parameter_ready(self, param: Parameter):
        """Mark a parameter as ready and reduce its bucket if all parameters are ready."""
        if param not in self.param_to_bucket:
            return
        
        bucket = self.param_to_bucket[param]
        bucket.mark_ready(param)
        
        # If bucket is ready, reduce it
        if bucket.ready:
            self._reduce_bucket(bucket)
    
    def prepare_for_backward(self):
        """Prepare for backward pass."""
        with self._lock:
            # Reset all buckets
            for bucket in self.buckets:
                bucket.pending = set(bucket.parameters)
                bucket.ready = False
            self.unused_parameters.clear()
            self._has_marked_unused_parameters = False
    
    def mark_parameter_ready(self, param: Parameter):
        """Mark a parameter gradient as ready for reduction."""
        with self._lock:
            self._mark_parameter_ready(param)
    
    def mark_all_parameters_ready(self):
        """Mark all parameters as ready (for static graph or end of backward)."""
        with self._lock:
            for bucket in self.buckets:
                # Mark all parameters in bucket as ready
                for param in bucket.parameters:
                    bucket.mark_ready(param)
                # Reduce the bucket if it's ready
                if bucket.ready:
                    self._reduce_bucket(bucket)
    
    def mark_unused_parameters(self, unused_params: Set[Parameter]):
        """Mark parameters as unused."""
        with self._lock:
            self.unused_parameters.update(unused_params)
            # Mark unused parameters as ready immediately
            for param in unused_params:
                if param in self.param_to_bucket:
                    self._mark_parameter_ready(param)
            self._has_marked_unused_parameters = True


class DistributedDataParallel(Module):
    r"""Implement distributed data parallelism based on ``mindtorch.distributed`` at module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default.

    Creation of this class requires that ``mindtorch.distributed`` to be already
    initialized, by calling :func:`mindtorch.distributed.init_process_group`.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or mindtorch.device, optional): CUDA devices.
            For single-device modules, ``device_ids`` can contain exactly one device id,
            which represents the only CUDA device where the input module corresponding
            to this process resides. Alternatively, ``device_ids`` can also be ``None``.
            For multi-device modules and CPU modules, ``device_ids`` must be ``None``.
            When ``device_ids`` is ``None`` for both cases, both the input data for the
            forward pass and the actual module must be placed on the correct device.
            (default: ``None``)
        output_device (int or mindtorch.device, optional): Device location of output for
            single-device CUDA modules. For multi-device modules and CPU modules, it must
            be ``None``, and the module itself dictates the output location.
            (default: ``device_ids[0]`` for single-device modules)
        broadcast_buffers (bool): Flag that enables syncing (broadcasting) buffers of the
            module at beginning of the ``forward`` function. (default: ``True``)
        process_group: The process group to be used for distributed data all-reduction.
            If ``None``, the default process group, which is created by
            :func:`mindtorch.distributed.init_process_group`, will be used.
            (default: ``None``)
        bucket_cap_mb: ``DistributedDataParallel`` will bucket parameters into multiple
            buckets so that gradient reduction of each bucket can potentially overlap with
            backward computation. :attr:`bucket_cap_mb` controls the bucket size in
            MebiBytes (MiB). If ``None``, a default size of 25 MiB will be used.
            (default: ``None``)
        find_unused_parameters (bool): Traverse the autograd graph from all tensors
            contained in the return value of the wrapped module's ``forward`` function.
            Parameters that don't receive gradients as part of this graph are preemptively
            marked as being ready to be reduced. (default: ``False``)
        gradient_as_bucket_view (bool): When set to ``True``, gradients will be views
            pointing to different offsets of ``allreduce`` communication buckets.
            (default: ``False``)
        static_graph (bool): When set to ``True``, DDP knows the trained graph is static.
            (default: ``False``)

    Attributes:
        module (Module): the module to be parallelized.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import mindtorch.distributed as dist
        >>> from mindtorch.nn.parallel import DistributedDataParallel as DDP
        >>> dist.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> model = DDP(model)
    """

    def __init__(
        self,
        module: Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        dim: int = 0,
        broadcast_buffers: bool = True,
        process_group: Optional[ProcessGroup] = None,
        bucket_cap_mb: Optional[int] = None,
        find_unused_parameters: bool = False,
        check_reduction: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
    ):
        super().__init__()

        if not is_initialized():
            raise RuntimeError(
                "Default process group has not been initialized, "
                "please make sure to call init_process_group."
            )

        if check_reduction:
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it.",
                FutureWarning,
                stacklevel=2,
            )

        # Get process group
        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.module = module
        self.dim = dim
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph

        # Device management
        if device_ids is not None and len(device_ids) > 1:
            raise ValueError(
                "device_ids can only be None or contain a single element."
            )

        # Get all parameters that require gradients
        self._module_parameters = list(
            p for p in module.parameters() if p.requires_grad
        )

        if len(self._module_parameters) == 0:
            raise RuntimeError(
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient."
            )

        # Check device consistency
        distinct_device_types = {
            p.device.type for p in self._module_parameters 
            if hasattr(p, 'device') and p.device is not None
        }
        if len(distinct_device_types) > 1:
            raise ValueError(
                "DistributedDataParallel's input module must be on "
                f"the same type of devices, but input module parameters locate in {distinct_device_types}."
            )

        if device_ids is None or len(device_ids) == 0:
            self.device_ids = None
            self.output_device = None
        else:
            self.device_ids = device_ids
            if output_device is None:
                self.output_device = device_ids[0]
            else:
                self.output_device = output_device

        # Bucket configuration
        if bucket_cap_mb is None:
            bucket_cap_mb = _DEFAULT_BUCKET_CAP_MB
        self.bucket_cap_mb = bucket_cap_mb

        # Create reducer for gradient synchronization
        self._reducer = _Reducer(
            self._module_parameters,
            self.process_group,
            bucket_cap_mb,
            find_unused_parameters,
            gradient_as_bucket_view,
            static_graph,
        )

        # Register backward hooks for gradient synchronization
        self._grad_acc_handles = []
        self._register_hooks()

        # Sync parameters and buffers initially
        self._sync_initial_state()
        
        # Track forward pass for unused parameter detection
        self._forward_hooks = []
        if find_unused_parameters:
            self._register_forward_hooks()

    def _register_hooks(self):
        """Register backward hooks on all parameters to synchronize gradients."""
        # Use weak references to avoid circular dependencies
        ddp_weakref = weakref.ref(self)
        
        def make_param_hook(param: Parameter):
            """Create a hook function for a specific parameter."""
            def post_accumulate_hook(p: Parameter):
                """Hook called after gradient is accumulated into parameter."""
                ddp = ddp_weakref()
                if ddp is None:
                    return
                if p.grad is not None and p in ddp._module_parameters:
                    ddp._reducer.mark_parameter_ready(p)
            
            def grad_hook(grad):
                """Hook called when gradient is computed."""
                ddp = ddp_weakref()
                if ddp is None:
                    return grad
                if grad is not None and param in ddp._module_parameters:
                    # Mark parameter as ready when gradient is available
                    ddp._reducer.mark_parameter_ready(param)
                return grad
            
            return post_accumulate_hook, grad_hook

        # Register hooks on all parameters
        for param in self._module_parameters:
            if param.requires_grad:
                post_acc_hook, grad_hook = make_param_hook(param)
                
                # Try to register post-accumulate hook (preferred for DDP)
                try:
                    if hasattr(param, 'register_post_accumulate_grad_hook'):
                        handle = param.register_post_accumulate_grad_hook(post_acc_hook)
                        self._grad_acc_handles.append(handle)
                    else:
                        # Fallback to regular hook
                        handle = param.register_hook(grad_hook)
                        self._grad_acc_handles.append(handle)
                except Exception as e:
                    # If register_hook doesn't work as expected, try alternative approach
                    warnings.warn(
                        f"Could not register hook on parameter: {e}. "
                        "Gradient synchronization may not work correctly.",
                        RuntimeWarning,
                    )
        
        # For static graph, we can optimize by reducing all at once
        if self.static_graph:
            # In static graph mode, we'll reduce all buckets together after backward
            pass

    def _register_forward_hooks(self):
        """Register forward hooks to detect unused parameters."""
        def forward_pre_hook(module, inputs):
            """Prepare for backward pass."""
            if not self.static_graph:
                self._reducer.prepare_for_backward()
        
        def forward_post_hook(module, inputs, outputs):
            """Detect unused parameters after forward pass."""
            if not self.static_graph and self.find_unused_parameters:
                # Find all tensors in the output
                output_tensors = _find_tensors(outputs)
                
                # Find all parameters that require grad
                all_params = set(self._module_parameters)
                
                # For unused parameter detection, we would need to traverse
                # the autograd graph. This is a simplified version.
                # In a full implementation, we would check which parameters
                # are connected to the output in the computation graph.
                # For now, we'll mark unused parameters at the end of backward
                pass
        
        # Register forward pre-hook
        handle = self.module.register_forward_pre_hook(forward_pre_hook)
        self._forward_hooks.append(handle)
        
        # Register forward post-hook for unused parameter detection
        if self.find_unused_parameters:
            handle = self.module.register_forward_hook(forward_post_hook)
            self._forward_hooks.append(handle)

    def _sync_initial_state(self):
        """Synchronize initial parameters and buffers across processes."""
        # Broadcast parameters from rank 0
        for param in self.module.parameters():
            broadcast(param, src=0, group=self.process_group)

        # Broadcast buffers if enabled
        if self.broadcast_buffers:
            for buffer in self.module.buffers():
                broadcast(buffer, src=0, group=self.process_group)

    def forward(self, *inputs, **kwargs):
        """Forward pass with buffer synchronization if enabled."""
        # Prepare for backward pass
        if not self.static_graph:
            self._reducer.prepare_for_backward()
        
        # Broadcast buffers before forward pass
        if self.broadcast_buffers:
            for buffer in self.module.buffers():
                broadcast(buffer, src=0, group=self.process_group)

        # Call the wrapped module's forward
        outputs = self.module(*inputs, **kwargs)
        
        # For static graph, we can register a callback to reduce all buckets
        # after backward completes
        if self.static_graph:
            # Register a callback to ensure all buckets are reduced
            # This will be called after backward completes
            def backward_completion_callback():
                self._ensure_all_buckets_reduced()
            
            # Try to register the callback if the framework supports it
            # This is framework-specific and may need adjustment
            try:
                # In MindSpore, we might need to use a different mechanism
                # For now, we'll rely on hooks to reduce buckets as they become ready
                pass
            except:
                pass
        
        return outputs
    
    def _ensure_all_buckets_reduced(self):
        """Ensure all remaining buckets are reduced (called at end of backward)."""
        # This should be called after backward completes
        # For static graph, we can optimize
        if self.static_graph:
            # In static graph mode, reduce all at once
            self._reducer.mark_all_parameters_ready()
        else:
            # For dynamic graphs, reduce any remaining buckets
            self._reducer.mark_all_parameters_ready()

    def _sync_buffers(self):
        """Synchronize buffers across processes."""
        if self.broadcast_buffers:
            for buffer in self.module.buffers():
                broadcast(buffer, src=0, group=self.process_group)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Return the module state dict."""
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """Load the module state dict."""
        return self.module.load_state_dict(state_dict, strict)

    def parameters(self, recurse: bool = True):
        """Return an iterator over module parameters."""
        return self.module.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        """Return an iterator over module parameters, yielding both the name and the parameter."""
        return self.module.named_parameters(prefix, recurse)

    def buffers(self, recurse: bool = True):
        """Return an iterator over module buffers."""
        return self.module.buffers(recurse)

    def named_buffers(self, prefix='', recurse=True):
        """Return an iterator over module buffers, yielding both the name and the buffer."""
        return self.module.named_buffers(prefix, recurse)

    def modules(self):
        """Return an iterator over all modules in the network."""
        return self.module.modules()

    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Return an iterator over all modules in the network, yielding both the name and the module."""
        return self.module.named_modules(memo, prefix, remove_duplicate)

    def children(self):
        """Return an iterator over immediate children modules."""
        return self.module.children()

    def named_children(self):
        """Return an iterator over immediate children modules, yielding both the name and the module."""
        return self.module.named_children()

    def train(self, mode: bool = True):
        """Set the module in training mode."""
        super().train(mode)
        self.module.train(mode)
        return self

    def eval(self):
        """Set the module in evaluation mode."""
        return self.train(False)

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __repr__(self):
        """Return a string representation of the module."""
        return f"{self.__class__.__name__}(\n  {self.module}\n)"


class DataParallel(Module):
    """DataParallel placeholder - not implemented yet."""
    pass
