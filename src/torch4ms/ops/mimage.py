import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops


def cubic_kernel(x, a=-0.75):
  """Cubic kernel with a = -0.75 (PyTorch-like Keys kernel)"""
  absx = mnp.abs(x)
  x2 = absx * absx
  x3 = x2 * absx
  cond1 = (absx <= 1)
  cond2 = (absx > 1) & (absx < 2)
  f1 = (a + 2) * x3 - (a + 3) * x2 + 1
  f2 = a * x3 - 5 * a * x2 + 8 * a * absx - 4 * a
  return ops.select(cond1, f1, ops.select(cond2, f2, 0.0))


def compute_contribs(in_size, out_size, scale, support=2.0, align_corners=False, dtype=None):
    """Compute indices and weights for interpolation"""
    if align_corners:
        if out_size == 1:
            in_coords = mnp.zeros((1,), dtype=dtype)
        else:
            in_coords = mnp.linspace(0, in_size - 1, out_size, dtype=dtype)
    else:
        out_coords = mnp.arange(out_size, dtype=dtype) + 0.5
        in_coords = out_coords / scale - 0.5

    left_idx = mnp.floor(in_coords).astype(ms.int32) - 1
    # 创建索引数组
    idxs = left_idx[:, None] + mnp.arange(4, dtype=ms.int32)

    dx = in_coords[:, None] - idxs.astype(dtype)
    weights = cubic_kernel(dx)

    # 归一化权重
    weights_sum = mnp.sum(weights, axis=1, keepdims=True)
    # 避免除零
    weights = weights / (weights_sum + 1e-12)
    return idxs, weights


def gather_weights(img, idxs, axis):
    """Safely gather with boundary handling"""
    # 裁剪索引到有效范围
    idxs = ops.clip_by_value(idxs, 0, img.shape[axis] - 1)
    # 使用gather操作
    return ops.gather(img, axis, idxs)


class InterpolateAlongAxisBCHW(ms.nn.Cell):
    """沿指定轴进行插值的Cell"""
    def __init__(self, axis):
        super(InterpolateAlongAxisBCHW, self).__init__()
        self.axis = axis
        self.clip = ops.clip_by_value
        self.take = ops.gather
        self.stack = ops.Stack(axis=0)
        self.tensordot = ops.tensordot
        self.moveaxis = ops.transpose

    def construct(self, img, idxs, weights):
        """
        Interpolate along H (axis=2) or W (axis=3) for tensor (B, C, H, W).
        idxs: (out_size, 4) int32 indices
        weights: (out_size, 4) float32 weights
        """
        assert self.axis in (2, 3), "Axis must be 2 (H) or 3 (W)"
        out_size = idxs.shape[0]
        k = idxs.shape[1]  # Typically 4 for cubic

        # 裁剪索引到输入边界
        idxs = self.clip(idxs, 0, img.shape[self.axis] - 1)  # (out_size, 4)

        # 初始化结果
        if self.axis == 2:
            # 插值H维度
            output_shape = (img.shape[0], img.shape[1], out_size, img.shape[3])
        else:
            # 插值W维度
            output_shape = (img.shape[0], img.shape[1], img.shape[2], out_size)
        output = ops.zeros(output_shape, dtype=img.dtype)

        # 对每个输出位置进行处理
        for i in range(out_size):
            idx = idxs[i]  # (4,)
            w = weights[i]  # (4,)
            weighted_sum = ops.zeros_like(img)

            # 收集并加权
            for o in range(k):
                gathered = self.take(img, self.axis, idx[o])
                weighted_sum += gathered * w[o]

            # 将结果放到正确位置
            if self.axis == 2:
                output[:, :, i, :] = weighted_sum
            else:
                output[:, :, :, i] = weighted_sum

        return output


def interpolate_bicubic_no_aa(img, out_h, out_w, align_corners=False):
    """双三次插值实现，不使用抗锯齿
    
    Args:
        img: 输入图像张量，形状为(B, C, H, W)
        out_h: 输出高度
        out_w: 输出宽度
        align_corners: 是否对齐角落像素
        
    Returns:
        插值后的图像张量
    """
    # 确保输入是MindSpore张量
    if not isinstance(img, ms.Tensor):
        img = ms.Tensor(img, dtype=ms.float32)
    
    h, w = img.shape[-2:]
    
    # 计算缩放比例
    if align_corners and out_h > 1:
        scale_y = (h - 1) / (out_h - 1)
    else:
        scale_y = out_h / h

    if align_corners and out_w > 1:
        scale_x = (w - 1) / (out_w - 1)
    else:
        scale_x = out_w / w

    # 计算y轴的贡献
    idxs_y, weights_y = compute_contribs(
        h,
        out_h,
        scale_y,
        align_corners=align_corners,
        dtype=img.dtype,
    )
    
    # 创建插值Cell
    interpolate_h = InterpolateAlongAxisBCHW(axis=2)
    interpolate_w = InterpolateAlongAxisBCHW(axis=3)
    
    # 先沿H轴插值
    tmp = interpolate_h(img, idxs_y, weights_y)

    # 计算x轴的贡献
    idxs_x, weights_x = compute_contribs(
        w,
        out_w,
        scale_x,
        align_corners=align_corners,
        dtype=img.dtype,
    )
    
    # 再沿W轴插值
    out = interpolate_w(tmp, idxs_x, weights_x)
    
    return out


def interpolate(img, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """MindSpore实现的图像插值函数，支持PyTorch的接口
    
    Args:
        img: 输入图像张量
        size: 输出大小 (H, W)
        scale_factor: 缩放因子
        mode: 插值模式，当前仅支持'nearest'和'bicubic'
        align_corners: 是否对齐角落像素
        
    Returns:
        插值后的图像张量
    """
    # 确保输入是MindSpore张量
    if not isinstance(img, ms.Tensor):
        img = ms.Tensor(img, dtype=ms.float32)
    
    # 计算输出大小
    if size is None:
        if scale_factor is None:
            raise ValueError("either size or scale_factor must be specified")
        h, w = img.shape[-2:]
        out_h = int(h * scale_factor)
        out_w = int(w * scale_factor)
    else:
        out_h, out_w = size
    
    # 处理align_corners默认值
    if align_corners is None:
        align_corners = False
    
    # 根据模式选择插值方法
    if mode == 'nearest':
        # 使用MindSpore内置的最近邻插值
        return ops.ResizeBilinear((out_h, out_w), align_corners=align_corners)(img)
    elif mode == 'bicubic':
        # 使用我们实现的双三次插值
        return interpolate_bicubic_no_aa(img, out_h, out_w, align_corners)
    else:
        raise NotImplementedError(f"Interpolation mode {mode} not implemented")