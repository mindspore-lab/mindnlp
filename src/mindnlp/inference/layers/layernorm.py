# FILE: nanovllm/layers/layernorm.py
from typing import Optional, Tuple, Union
import mindtorch
from mindtorch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(mindtorch.ones(hidden_size))

    @mindtorch.compile
    def rms_forward(
        self,
        x: mindtorch.Tensor,
    ) -> mindtorch.Tensor:
        orig_dtype = x.dtype
        x = x.to(mindtorch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(mindtorch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @mindtorch.compile
    def add_rms_forward(
        self,
        x: mindtorch.Tensor,
        residual: mindtorch.Tensor,
    ) -> tuple[mindtorch.Tensor, mindtorch.Tensor]:
        orig_dtype = x.dtype
        x = x.to(mindtorch.float32).add_(residual.to(mindtorch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(mindtorch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: mindtorch.Tensor,
        residual: Optional[mindtorch.Tensor] = None,
    ) -> Union[mindtorch.Tensor, Tuple[mindtorch.Tensor, mindtorch.Tensor]]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
