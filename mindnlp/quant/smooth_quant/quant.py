"""quant implement"""
from typing import Optional, Tuple
import mindspore
from mindspore import Tensor
from mindspore.ops.primitive import PrimitiveWithInfer, prim_attr_register
from mindnlp.core import nn, ops
from mindnlp.core.serialization import load
from mindnlp.configs import ON_ORANGE_PI

from .smooth import smooth_lm


class BatchMatMulV2(PrimitiveWithInfer):
    @prim_attr_register
    def __init__(self, transpose_a=False, transpose_b=False): # pylint: disable=super-init-not-called
        """Initialize BatchMatMul."""
        self.init_prim_io_names(inputs=["x1", "x2", "bias", "offset_w"], outputs=["y"])
        self.add_prim_attr("adj_x1", self.transpose_a)
        self.add_prim_attr("adj_x2", self.transpose_b)

    def infer_shape(self, x1_shape, x2_shape, bias_shape=None):
        return x1_shape[:-1] + [x2_shape[-1]]

    def infer_dtype(self, x_dtype, v_dtype, bias_dtype=None):
        if x_dtype == mindspore.TensorType(mindspore.int8):
            return mindspore.TensorType(mindspore.int32)
        return x_dtype

matmulInteger = BatchMatMulV2()


def quantize_mat(mat: Tensor) -> Tuple[Tensor, Tensor]:
    max_val = (ops.max(ops.abs(mat), dim=-1)[0] / 127.0).to(dtype=mat.dtype)
    mat = (mat / max_val[..., None]).to(dtype=mindspore.int8)
    return mat, max_val


def dequantize_mat(mat: Tensor, max_val: Tensor):
    return ops.mul(mat, max_val.unsqueeze(-1))


def decomposition(mat: Tensor, unq_idx: Tensor, t: Tensor):
    return mat.mul(t.to(dtype=mat.dtype)), mat[..., unq_idx]
    # mat = mat.clone()
    # mat_unq = mat[..., unq_idx]
    # if mat.dim() == 3:
    #     mat[:, :, unq_idx] = 0
    # elif mat.dim() == 4:
    #     mat[:, :, :, unq_idx] = 0
    # elif mat.dim() == 2:
    #     mat[:, unq_idx] = 0
    # return mat, mat_unq


def get_unq_idx_topk(mat: Tensor, k: int = 64):
    idx = ops.topk(ops.max(mat.view(-1, mat.shape[-1]).abs(), dim=-2)[0], k, dim=-1)[1]
    t = ops.ones((mat.shape[-1]), dtype=mat.dtype)
    t = t.copy()
    if ON_ORANGE_PI:
        ops.setitem(t, idx, 0)
    else:
        t[idx] = 0
    return idx, t


def get_unq_idx_thres(mat: Tensor, threshold: float = 6.0):
    k = ops.max(mat.view(-1, mat.shape[-1]).abs(), dim=-2)[0] >= threshold
    return ops.nonzero(k).view(-1), k


def qMatmul(x_q: Tensor, x_max: Tensor, weight_q: Tensor, w_max: Tensor, dtype):
    res_q = matmulInteger(x_q, weight_q)
    mx = nn.functional.linear(x_max.unsqueeze(-1), w_max.unsqueeze(-1))
    res = ops.mul(res_q.to(dtype=mindspore.float32), mx.to(mindspore.float32)).to(dtype=dtype)
    return res


class W8Linear(nn.Module):
    def __init__(
        self,
        origin_weight: Tensor,
        bias: Optional[Tensor] = None,
        act_max: Optional[Tensor] = None,
        alpha=32,
    ):
        super().__init__()
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.dtype = origin_weight.dtype
        self.alpha = alpha
        self.weight_q, self.max_val = quantize_mat(origin_weight)
        self.weight_q = nn.Parameter(self.weight_q, requires_grad=False)
        self.max_val = nn.Parameter(self.max_val, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(
            x, dequantize_mat(self.weight_q, self.max_val), bias=self.bias
        )


# act_max for smooth
class W8X8Linear(nn.Module):
    def __init__(
        self,
        ori_w: Tensor,
        bias: Optional[Tensor] = None,
        act_max: Optional[Tensor] = None,
        alpha=32,
    ):
        super().__init__()
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.dtype = ori_w.dtype
        self.alpha = alpha
        self.scales = None
        if act_max is not None:
            self.scales = (
                (act_max.pow(alpha) / ops.max(ori_w.abs(), dim=0)[0].pow(1 - alpha))
                .clamp(min=1e-5)
                .to(dtype=ori_w.dtype)
            )
            self.scales = nn.Parameter(self.scales, requires_grad=False)
            ori_w = ori_w.mul(self.scales)
        self.weight_q, self.max_val = quantize_mat(ori_w)
        self.weight_q = nn.Parameter(self.weight_q.t(), requires_grad=False)
        self.max_val = nn.Parameter(self.max_val, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.scales is not None:
            x = x.div(self.scales)
        x_q, x_max = quantize_mat(x)
        res = qMatmul(x_q, x_max, self.weight_q, self.max_val, x.dtype)
        if self.bias is not None:
            res = res + self.bias
        return res


# static decomposition
class W8SDLinear(nn.Module):
    def __init__(
        self,
        origin_weight: Tensor,
        bias: Optional[Tensor] = None,
        act_max: Optional[Tensor] = None,
        alpha=32,
    ):
        super().__init__()
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.dtype = origin_weight.dtype
        self.alpha = alpha
        if act_max is not None:
            self.idx_unq, self.t = get_unq_idx_topk(act_max, self.alpha)
        else:
            self.idx_unq, self.t = get_unq_idx_topk(origin_weight, self.alpha)

        self.weight_q, self.weight_unq = decomposition(
            origin_weight, self.idx_unq, self.t
        )
        self.weight_q, self.w_max = quantize_mat(self.weight_q)
        self.weight_q = nn.Parameter(self.weight_q.t(), requires_grad=False)
        self.weight_unq = nn.Parameter(self.weight_unq.t(), requires_grad=False)
        self.w_max = nn.Parameter(self.w_max, requires_grad=False)
        self.t = nn.Parameter(self.t, requires_grad=False)
        self.idx_unq = nn.Parameter(self.idx_unq, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        x_q, x_unq = decomposition(x, self.idx_unq, self.t)
        x_q, x_max = quantize_mat(x_q)
        res_q = qMatmul(x_q, x_max, self.weight_q, self.w_max, x.dtype)
        res_unq = ops.matmul(x_unq, self.weight_unq)
        if self.bias is not None:
            res_unq += self.bias
        return res_q + res_unq


class W8DXLinear(nn.Module):
    def __init__(
        self,
        origin_weight: Tensor,
        bias: Optional[Tensor] = None,
        act_max: Optional[Tensor] = None,
        alpha=32,
    ):
        super().__init__()
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.dtype = origin_weight.dtype
        self.alpha = alpha
        self.weight_q, self.max_val = quantize_mat(origin_weight)
        self.weight_q = nn.Parameter(self.weight_q.t(), requires_grad=False)
        self.max_val = nn.Parameter(self.max_val, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        idx_unq, t = get_unq_idx_topk(x, self.alpha)
        x_q, x_unq = decomposition(x, idx_unq, t)
        x_q, x_max = quantize_mat(x_q)
        res_q = qMatmul(x_q, x_max, self.weight_q, self.max_val, x.dtype)
        weight_unq = ops.mul(self.weight_q[idx_unq, :], self.max_val.unsqueeze(0))
        res_unq = ops.matmul(x_unq, weight_unq)
        if self.bias is not None:
            res_unq += self.bias
        return res_q + res_unq


quant_cls = {"W8": W8Linear, "W8X8": W8X8Linear, "W8SD": W8SDLinear, "W8DX": W8DXLinear}


def replace_linear_modules(module: nn.Module, prefix: str, act_scales, cfg):
    for name, child in module.named_children():
        fullname = (prefix + "." + name) if prefix != "" else name
        if isinstance(child, nn.Linear):
            strs = fullname.split(".")
            # fullname: model.layers.21.self_attn.q_proj layer_name: 21.q_proj; name: q_proj
            # fullname: lm_head; layer_name: 21.q_proj; name: q_proj;
            layer_name = (strs[-3] + "." + strs[-1]) if len(strs) > 2 else strs[-1]

            if layer_name not in cfg:
                continue
            act_scale = (
                None
                if act_scales is None or "act_scale" not in cfg[layer_name]
                else act_scales[fullname]
            )
            alpha = None if "alpha" not in cfg[layer_name] else cfg[layer_name]["alpha"]
            setattr(
                module,
                name,
                quant_cls[cfg[layer_name]["type"]](
                    child.weight, child.bias, act_max=act_scale, alpha=alpha
                ),
            )
        else:
            replace_linear_modules(child, fullname, act_scales, cfg)


def quantize(model: nn.Module, cfg={}):
    act_scales = None
    if "act_scales_path" in cfg:
        act_scales = load(cfg["act_scales_path"])
        if "smooth" in cfg:

            alpha = 0.85 if "alpha" not in cfg else cfg["alpha"]
            smooth_lm(model, act_scales, alpha)
    replace_linear_modules(model, "", act_scales, cfg)
