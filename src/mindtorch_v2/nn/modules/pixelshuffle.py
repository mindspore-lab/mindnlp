from ..module import Module
from .. import functional as F


class PixelShuffle(Module):
    """Rearranges elements in a tensor of shape (N, C*r^2, H, W) to (N, C, H*r, W*r).

    Args:
        upscale_factor (int): factor to increase spatial resolution by.
    """

    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self):
        return f'upscale_factor={self.upscale_factor}'


class PixelUnshuffle(Module):
    """Reverses the PixelShuffle operation: (N, C, H*r, W*r) -> (N, C*r^2, H, W).

    Args:
        downscale_factor (int): factor to decrease spatial resolution by.
    """

    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self):
        return f'downscale_factor={self.downscale_factor}'


class ChannelShuffle(Module):
    """Divides channels into groups and shuffles them to mix information across groups.

    Args:
        groups (int): number of groups to divide channels into.
    """

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, input):
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self):
        return f'groups={self.groups}'
