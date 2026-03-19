import math
import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import nn, ops, Tensor, Parameter

from utils.ops import dist2bbox
from utils.tal import make_anchors


# 基础工具函数与算子

def autopad(k, p=None, d=1):
    """自动计算填充 (Padding) 以保持输出特征图尺寸一致"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Identity(nn.Cell):
    """恒等映射层"""
    def construct(self, x):
        return x

class Concat(nn.Cell):
    """张量拼接层"""
    def __init__(self, axis=1):
        super(Concat, self).__init__()
        self.axis = axis
    def construct(self, x):
        return ops.concat(x, self.axis)

class Upsample(nn.Cell):
    """
    上采样模块
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def construct(self, x):
        if self.scale_factor is not None:
            shape = x.shape
            new_size = (int(shape[-2] * self.scale_factor), int(shape[-1] * self.scale_factor))
            return ops.interpolate(x, size=new_size, mode=self.mode)
        
        return ops.interpolate(x, size=self.size, mode=self.mode)


# 核心卷积模块

class ConvNormAct(nn.Cell):
    """标准卷积块：包含 Conv2d + BatchNorm2d + 激活函数"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, momentum=0.97, eps=1e-3, sync_bn=False):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, pad_mode="pad", padding=autopad(k, p, d), group=g, dilation=d, has_bias=False
        )
        self.bn = nn.BatchNorm2d(c2, momentum=momentum, eps=eps).to_float(ms.float32)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Cell) else Identity())

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(ConvNormAct):
    """深度可分离卷积 (Depthwise Convolution)"""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True, sync_bn=False):
        # 使用 math.gcd 确保 group 数等于输入输出的最大公约数
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act, sync_bn=sync_bn)

class Bottleneck(nn.Cell):
    """标准瓶颈块 (Bottleneck)"""
    def __init__(self, c1, c2, shortcut=True, k=(3, 3), g=(1, 1), e=0.5, act=True, momentum=0.97, eps=1e-3, sync_bn=False):
        super().__init__()
        c_ = int(c2 * e)
        # 根据官方设计，此处强制设置 group=1
        self.conv1 = ConvNormAct(c1, c_, k[0], 1, g=1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.conv2 = ConvNormAct(c_, c2, k[1], 1, g=1, act=act, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.add = shortcut and c1 == c2
        
    def construct(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C2f(nn.Cell):
    """C2f 模块 (CSP Bottleneck with 2 convolutions)"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, momentum=0.97, eps=1e-3, sync_bn=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.cv2 = ConvNormAct((2 + n) * self.c, c2, 1, momentum=momentum, eps=eps, sync_bn=sync_bn)
        self.m = nn.CellList([Bottleneck(self.c, self.c, shortcut, k=(3, 3), g=(1, g), e=1.0, 
                                        momentum=momentum, eps=eps, sync_bn=sync_bn) for _ in range(n)])
    def construct(self, x):
        x = self.cv1(x)
        y = list(ops.split(x, axis=1, split_size_or_sections=self.c))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(ops.concat(y, axis=1))

class C3k(nn.Cell):
    """C3k 模块"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3, sync_bn=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(c1, c_, 1, 1, sync_bn=sync_bn)
        self.cv3 = ConvNormAct(2 * c_, c2, 1, sync_bn=sync_bn)
        self.m = nn.SequentialCell([Bottleneck(c_, c_, shortcut, k=(k, k), g=1, e=1.0, sync_bn=sync_bn) for _ in range(n)])
        
    def construct(self, x):
        return self.cv3(ops.concat((self.m(self.cv1(x)), self.cv2(x)), axis=1))

class C3k2(C2f):
    """YOLO11 核心模块 C3k2 (基于 C2f 优化的 C3k 变体)"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, sync_bn=False):
        super().__init__(c1, c2, n, shortcut, g, e, sync_bn=sync_bn)
        self.m = nn.CellList([
            C3k(self.c, self.c, 2, shortcut, g, sync_bn=sync_bn) if c3k else 
            Bottleneck(self.c, self.c, shortcut, k=(3, 3), g=(1, g), sync_bn=sync_bn) for _ in range(n)
        ])

class SPPF(nn.Cell):
    """空间金字塔池化模块 (Spatial Pyramid Pooling - Fast)"""
    def __init__(self, c1, c2, k=5, momentum=0.97, eps=1e-3):
        super(SPPF, self).__init__()
        c_ = c1 // 2 
        self.cv1 = ConvNormAct(c1, c_, 1, 1, momentum=momentum, eps=eps)
        self.cv2 = ConvNormAct(c_ * 4, c2, 1, 1, momentum=momentum, eps=eps)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, pad_mode="same")

    def construct(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(ops.concat((x, y1, y2, y3), 1))


# 注意力机制模块

class Attention(nn.Cell):
    """多头自注意力模块 (Multi-Head Self Attention)"""
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvNormAct(c1=dim, c2=h, k=1, act=False)
        self.proj = ConvNormAct(c1=dim, c2=dim, k=1, act=False)
        self.pe = ConvNormAct(c1=dim, c2=dim, k=3, s=1, g=dim, act=False)

    def construct(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], axis=2)
        attn = (ops.transpose(q, (0, 1, 3, 2)) @ k) * self.scale
        attn = ops.softmax(attn, axis=-1)
        x = (v @ ops.transpose(attn, (0, 1, 3, 2))).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)

class PSABlock(nn.Cell):
    """PSA (Polarized Self-Attention) 模块"""
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, sync_bn=False):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.SequentialCell([
            ConvNormAct(c, c * 2, 1, sync_bn=sync_bn), 
            ConvNormAct(c * 2, c, 1, act=False, sync_bn=sync_bn)
        ])
        self.add = shortcut
    def construct(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

class C2PSA(nn.Cell):
    """融合 PSA 注意力机制的 C2 模块"""
    def __init__(self, c1, c2, n=1, e=0.5, sync_bn=False):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvNormAct(c1, 2 * self.c, 1, 1, sync_bn=sync_bn)
        self.cv2 = ConvNormAct(2 * self.c, c1, 1, sync_bn=sync_bn)
        self.m = nn.SequentialCell([PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64, sync_bn=sync_bn) for _ in range(n)])
    def construct(self, x):
        x = self.cv1(x)
        a, b = ops.split(x, axis=1, split_size_or_sections=self.c)
        return self.cv2(ops.concat((a, self.m(b)), 1))


# 网络头部模块 (Heads)

class Classify(nn.Cell):
    """YOLO11 分类任务头部"""
    def __init__(self, c1, c2, *args, **kwargs):
        super().__init__()
        c_ = 1280  # efficientnet_b0 默认输出尺寸
        
        # 强制设置 k=1, s=1 确保拓扑结构一致
        self.conv = ConvNormAct(c1, c_, k=1, s=1) 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.0)
        self.linear = nn.Dense(c_, c2)

    def construct(self, x):
        if isinstance(x, (list, tuple)):
            x = ops.concat(x, 1)
            
        x = self.conv(x)
        x = self.pool(x)
        x = ops.flatten(x, start_dim=1)
        x = self.drop(x)
        x = self.linear(x)
        
        if self.training:
            return x
        y = ops.softmax(x, axis=1) 
        return y, x

class DFL(nn.Cell):
    """分布焦点损失 (Distribution Focal Loss) 积分计算模块"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, has_bias=False)
        self.conv.weight.requires_grad = False  # 积分序列权重应保持固定
        self.c1 = c1
        self.softmax = ops.Softmax(axis=1)
        self.initialize_conv_weight()

    def initialize_conv_weight(self):
        """初始化离散积分权重"""
        self.conv.weight.set_data(
            Tensor(np.arange(self.c1).reshape((1, self.c1, 1, 1)), dtype=ms.float32)
        )

    def construct(self, x):
        s = x.shape
        b, c, a = s[0], s[1], s[-1]
        x = self.softmax(x.view(b, 4, self.c1, a).swapaxes(2, 1))
        x = self.conv(x)
        return x.view(b, 4, a)

class YOLO11DetectHead(nn.Cell):
    """YOLO11 目标检测头部"""
    def __init__(self, nc=80, reg_max=16, stride=(), ch=(), sync_bn=False):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4 
        
        self.stride = Parameter(ms.Tensor(stride, ms.int32), requires_grad=False)
        
        c2 = max((16, ch[0] // 4, reg_max * 4))
        c3 = max(ch[0], min(nc, 100))
        
        self.cv2 = nn.CellList([
            nn.SequentialCell([
                ConvNormAct(x, c2, 3), 
                ConvNormAct(c2, c2, 3), 
                nn.Conv2d(c2, 4 * reg_max, 1, has_bias=True)
            ]) for x in ch
        ])
        
        self.cv3 = nn.CellList([
            nn.SequentialCell([
                nn.SequentialCell([DWConv(x, x, 3), ConvNormAct(x, c3, 1)]),
                nn.SequentialCell([DWConv(c3, c3, 3), ConvNormAct(c3, c3, 1)]),
                nn.Conv2d(c3, nc, 1, has_bias=True, pad_mode="pad", padding=0)
            ]) for x in ch
        ])
        self.dfl = DFL(reg_max)

    def construct(self, x):
        """
        YOLO11 检测头前向传播
        训练模式：返回 (拼接后的展平Tensor, 原始特征图List)
        推理模式：返回 (最终预测结果, 原始特征图List)
        """
        # 1. 基础卷积处理
        res = []
        for i in range(self.nl):
            # 保持 [B, 64+nc, H, W] 结构
            res.append(ops.concat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))

        # 2. 统一展平逻辑 (训练和推理都需要这个展平后的 x_cat 来算 Loss)
        y = []
        for xi in res:
            bs, ch, h, w = xi.shape
            y.append(xi.view(bs, ch, -1))
        
        # 拼接所有尺度：[B, 144, 19200]
        x_cat = ops.concat(y, 2)
        
        if self.training:
            return x_cat, res

        # 3. 推理解码逻辑 (仅在 self.training=False 时执行)
        box_dist, cls_logits = ops.split(x_cat, (self.reg_max * 4, self.nc), axis=1)
        
        # DFL 积分回归
        box_decoded = self.dfl(box_dist) # 形状 [B, 4, 19200]
        
        # 获取锚点和步长
        # 这里的 x 必须是原始特征图列表
        anchors, strides = self.make_anchors(x, self.stride, 0.5)
        
        # 解码坐标：直接传入原始 anchors，利用 ops.py 里的自适应逻辑
        dbox = dist2bbox(box_decoded, anchors, strides=strides, xywh=True, axis=1)

        # 最终拼接：[B, 4+nc, 19200]
        final_pred = ops.concat((dbox, ops.sigmoid(cls_logits)), 1)
        
        return final_pred, res


class YOLO11Pose(YOLO11DetectHead):
    """YOLO11 姿态估计头部"""
    def __init__(self, nc=1, kpt_shape=(17, 3), reg_max=16, stride=(), ch=(), sync_bn=False):
        nc = 1
        super().__init__(nc, reg_max, stride, ch, sync_bn)
        self.kpt_shape = kpt_shape
        self.nkpt = kpt_shape[0] * kpt_shape[1]
        c4 = max(ch[0] // 4, self.nkpt)
        
        self.cv4 = nn.CellList([
            nn.SequentialCell([
                ConvNormAct(x, c4, 3),
                ConvNormAct(c4, c4, 3),
                nn.Conv2d(c4, self.nkpt, 1, has_bias=True)
            ]) for x in ch
        ])

    def kpts_decode(self, kpt_flat, anc, strides):
        """解析并还原关键点坐标及其可见度"""
        B = kpt_flat.shape[0]
        
        # 统一提升至 float32 精度，防止混合精度计算时发生溢出
        kpt_flat = kpt_flat.astype(ms.float32)
        anc = anc.astype(ms.float32)
        strides = strides.astype(ms.float32)
        
        kpt = kpt_flat.transpose(0, 2, 1).view(B, -1, self.kpt_shape[0], self.kpt_shape[1])
        
        anc_safe = anc.view(1, -1, 1, 2)
        strides_safe = strides.view(1, -1, 1, 1)
        
        xy = kpt[..., :2]
        xy_decoded = (xy * 2.0 + (anc_safe - 0.5)) * strides_safe
        vis = ops.sigmoid(kpt[..., 2:3])
        
        kpt_decoded = ops.concat((xy_decoded, vis), axis=-1).view(B, -1, self.nkpt)
        return kpt_decoded.transpose(0, 2, 1)

    def construct(self, x):
        bs = x[0].shape[0]
        box_outs, cls_outs, kpt_outs = [], [], []
        for i in range(self.nl):
            box_outs.append(self.cv2[i](x[i]))
            cls_outs.append(self.cv3[i](x[i]))
            kpt_outs.append(self.cv4[i](x[i]))

        if self.training:
            out = tuple(ops.concat((box_outs[i], cls_outs[i]), 1) for i in range(self.nl))
            return out, kpt_outs

        box_flat = ops.concat([b.view(bs, self.reg_max * 4, -1) for b in box_outs], axis=2)
        cls_flat = ops.concat([c.view(bs, self.nc, -1) for c in cls_outs], axis=2)
        kpt_flat = ops.concat([k.view(bs, self.nkpt, -1) for k in kpt_outs], axis=2)

        # 统一提升精度以保证解码阶段数值稳定性
        box_flat = box_flat.astype(ms.float32)
        cls_flat = cls_flat.astype(ms.float32)
        kpt_flat = kpt_flat.astype(ms.float32)

        cls_prob = ops.sigmoid(cls_flat) 

        self.anchors, self.strides = make_anchors(x, self.stride, 0.5)
        anc = self.anchors.astype(ms.float32).view(1, -1, 2).transpose(0, 2, 1)   
        strd = self.strides.astype(ms.float32).view(1, -1, 1).transpose(0, 2, 1)  

        box_dist = self.dfl(box_flat)
        lt = box_dist[:, :2, :]
        rb = box_dist[:, 2:, :]
        c_xy = anc + (rb - lt) / 2.0
        wh = rb + lt
        dbox = ops.concat((c_xy, wh), axis=1) * strd 

        pred_kpt = self.kpts_decode(kpt_flat, self.anchors, self.strides)

        # 按照 (Box, Cls, Pose) 顺序拼接
        y = ops.concat((dbox, cls_prob, pred_kpt), axis=1) 
        return y.transpose(0, 2, 1)
        
class ProtoCell(nn.Cell):
    """实例分割任务中的原型 (Prototype) 生成模块"""
    def __init__(self, c1, c_=256, c2=32, sync_bn=False):
        super().__init__()
        self.cv1 = ConvNormAct(c1, c_, 3, sync_bn=sync_bn, momentum=0.9, eps=1e-3)
        self.upsample = nn.Conv2dTranspose(c_, c_, 2, 2, has_bias=True) 
        self.cv2 = ConvNormAct(c_, c_, 3, sync_bn=sync_bn, momentum=0.9, eps=1e-3)
        self.cv3 = ConvNormAct(c_, c2, k=1, sync_bn=sync_bn, act=False)

    def construct(self, x):
        x = self.cv1(x)
        x = self.upsample(x)
        x = self.cv2(x)
        x = self.cv3(x)
        
        # 注意：此处限制数值范围旨在避免混合精度下的异常溢出
        # 规范做法建议排查底层 float16 乘加运算，或依赖全局 FP32 转换
        x = ops.clip_by_value(x, -10.0, 10.0) 
        return x

class YOLO11Segment(YOLO11DetectHead):
    """YOLO11 实例分割头部"""
    def __init__(self, nc=80, reg_max=16, nm=32, npr=256, stride=(), ch=(), sync_bn=False):
        super().__init__(nc=nc, reg_max=reg_max, stride=stride, ch=ch, sync_bn=sync_bn)
        self.nm = nm
        c4 = max(ch[0] // 4, nm)
        _npr = max(c4, min(ch[0], npr))
        self.proto = ProtoCell(ch[0], _npr, nm, sync_bn=sync_bn)
        
        self.cv4 = nn.CellList([
            nn.SequentialCell([
                ConvNormAct(x, c4, 3, sync_bn=sync_bn),
                ConvNormAct(c4, c4, 3, sync_bn=sync_bn),
                nn.Conv2d(c4, nm, 1, has_bias=True)
            ]) for x in ch
        ])

    def construct(self, x):
        p_out = self.proto(x[0])
        
        bs = x[0].shape[0]
        out, mc = (), () 
        for i in range(self.nl):
            xi = x[i]
            cv2_feat = self.cv2[i](xi)
            cv3_feat = self.cv3[i](xi)
            
            out += (ops.concat((cv2_feat, cv3_feat), 1),)
            mc += (self.cv4[i](xi).view(bs, self.nm, -1),)

        if self.training:
            return out, ops.concat(mc, 2), p_out
        
        self.anchors, self.strides = self.make_anchors(out, self.stride, 0.5)
        x_all = ops.concat([xi.view(bs, self.no, -1) for xi in out], 2)
        
        box, cls = ops.split(x_all, (self.reg_max * 4, self.nc), 1)
        
        box_decoded = self.dfl(box) 
        box_decoded = ops.relu(box_decoded)
        dbox = dist2bbox(box_decoded.transpose(0, 2, 1), 
                              ops.expand_dims(self.anchors, 0), 
                              self.strides, xywh=True)
        
        mc_all = ops.concat(mc, 2)
        final_pred = ops.concat((dbox, ops.sigmoid(cls), mc_all), 1)
        
        return final_pred, p_out