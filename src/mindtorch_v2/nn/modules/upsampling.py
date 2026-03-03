from ..module import Module
from .. import functional as F


class Upsample(Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest',
                 align_corners=None, recompute_scale_factor=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode,
                             self.align_corners, self.recompute_scale_factor)

    def extra_repr(self):
        if self.scale_factor is not None:
            return f'scale_factor={self.scale_factor}, mode={repr(self.mode)}'
        return f'size={self.size}, mode={repr(self.mode)}'
