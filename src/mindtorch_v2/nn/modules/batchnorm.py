"""Batch normalization modules."""

import numpy as np
from ..module import Module
from ..parameter import Parameter
import mindtorch_v2 as torch


class _BatchNorm(Module):
    """Base class for batch normalization layers."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.int64))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean = torch.zeros(self.num_features)
            self.running_var = torch.ones(self.num_features)
            self.num_batches_tracked = torch.tensor(0, dtype=torch.int64)

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight = Parameter(torch.ones(self.num_features))
            self.bias = Parameter(torch.zeros(self.num_features))

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)
        return self._batch_norm(input)

    def _batch_norm(self, input):
        """Apply batch normalization."""
        x = input.numpy()

        if self.training and self.track_running_stats:
            # Calculate statistics over batch and spatial dimensions
            axes = self._get_reduction_axes(x.ndim)
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)

            # Update running statistics
            if self.running_mean is not None:
                running_mean = self.running_mean.numpy()
                running_var = self.running_var.numpy()
                running_mean = (1 - self.momentum) * running_mean + self.momentum * mean.flatten()
                running_var = (1 - self.momentum) * running_var + self.momentum * var.flatten()
                self.running_mean = torch.tensor(running_mean)
                self.running_var = torch.tensor(running_var)
        else:
            # Use running statistics
            if self.running_mean is not None:
                mean = self.running_mean.numpy().reshape(self._get_reshape_dims(x.ndim))
                var = self.running_var.numpy().reshape(self._get_reshape_dims(x.ndim))
            else:
                axes = self._get_reduction_axes(x.ndim)
                mean = np.mean(x, axis=axes, keepdims=True)
                var = np.var(x, axis=axes, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        if self.affine:
            weight = self.weight.numpy().reshape(self._get_reshape_dims(x.ndim))
            bias = self.bias.numpy().reshape(self._get_reshape_dims(x.ndim))
            x_norm = x_norm * weight + bias

        return torch.tensor(x_norm.astype(np.float32))

    def _get_reduction_axes(self, ndim):
        raise NotImplementedError

    def _get_reshape_dims(self, ndim):
        raise NotImplementedError


class BatchNorm1d(_BatchNorm):
    """Batch normalization for 2D or 3D input (N, C) or (N, C, L)."""

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input, got {input.dim()}D")

    def _get_reduction_axes(self, ndim):
        if ndim == 2:
            return (0,)  # (N, C)
        return (0, 2)  # (N, C, L)

    def _get_reshape_dims(self, ndim):
        if ndim == 2:
            return (1, -1)
        return (1, -1, 1)


class BatchNorm2d(_BatchNorm):
    """Batch normalization for 4D input (N, C, H, W)."""

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input, got {input.dim()}D")

    def _get_reduction_axes(self, ndim):
        return (0, 2, 3)  # (N, C, H, W)

    def _get_reshape_dims(self, ndim):
        return (1, -1, 1, 1)


class BatchNorm3d(_BatchNorm):
    """Batch normalization for 5D input (N, C, D, H, W)."""

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError(f"Expected 5D input, got {input.dim()}D")

    def _get_reduction_axes(self, ndim):
        return (0, 2, 3, 4)  # (N, C, D, H, W)

    def _get_reshape_dims(self, ndim):
        return (1, -1, 1, 1, 1)


class SyncBatchNorm(BatchNorm2d):
    """Synchronized batch normalization - falls back to BatchNorm2d."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats,
                        device, dtype)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """Convert BatchNorm to SyncBatchNorm - returns module unchanged."""
        return module
