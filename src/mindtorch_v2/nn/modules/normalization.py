from ..module import Module
from ..parameter import Parameter
from ..._creation import tensor
from .. import functional as F


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(tensor([1.0] * self.normalized_shape[-1]))
            if bias:
                self.bias = Parameter(tensor([0.0] * self.normalized_shape[-1]))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(tensor([1.0] * num_features))
            self.bias = Parameter(tensor([0.0] * num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if track_running_stats:
            self.register_buffer('running_mean', tensor([0.0] * num_features))
            self.register_buffer('running_var', tensor([1.0] * num_features))
            self.register_buffer('num_batches_tracked', tensor([0.0]))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var,
                            self.weight, self.bias, self.training, self.momentum, self.eps)

    def extra_repr(self):
        return (f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, '
                f'affine={self.affine}, track_running_stats={self.track_running_stats}')


class BatchNorm2d(BatchNorm1d):
    pass


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, device=None, dtype=None):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Parameter(tensor([1.0] * num_channels))
            self.bias = Parameter(tensor([0.0] * num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return f'{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}'


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(tensor([1.0] * self.normalized_shape[-1]))
        else:
            self.register_parameter('weight', None)

    def forward(self, input):
        raise NotImplementedError("RMSNorm forward is not yet implemented")

    def extra_repr(self):
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
