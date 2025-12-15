import mindtorch
from mindtorch import nn
import mindtorch.nn.functional as F


class SiluAndMul(nn.Module):
    @mindtorch.compile
    def forward(self, x: mindtorch.Tensor) -> mindtorch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
