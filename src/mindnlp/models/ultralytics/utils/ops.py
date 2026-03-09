import math
import numpy as np
import mindspore as ms
from mindspore import ops

def xyxy2xywh(x):
    """边界框格式转换：(x1, y1, x2, y2) -> (xc, yc, w, h)"""
    y = ops.deepcopy(x) if isinstance(x, ms.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  
    y[..., 2] = x[..., 2] - x[..., 0]  
    y[..., 3] = x[..., 3] - x[..., 1]  
    return y


def xywh2xyxy(x):
    """边界框格式转换：(xc, yc, w, h) -> (x1, y1, x2, y2)"""
    y = ops.deepcopy(x) if isinstance(x, ms.Tensor) else np.copy(x)
    
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """归一化相对坐标转为物理像素坐标"""
    y = ops.deepcopy(x) if isinstance(x, ms.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  
    return y


def clip_boxes(boxes, shape):
    """将边界框坐标截断至图像物理边界内"""
    h, w = shape[:2]
    if isinstance(boxes, ms.Tensor):
        boxes[..., 0] = ops.clamp(boxes[..., 0], 0, w)
        boxes[..., 1] = ops.clamp(boxes[..., 1], 0, h)
        boxes[..., 2] = ops.clamp(boxes[..., 2], 0, w)
        boxes[..., 3] = ops.clamp(boxes[..., 3], 0, h)
    else:
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, w)
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, h)
    return boxes


def segment2box(segment, width=640, height=640):
    """由不规则多边形轮廓 (Segments) 推导其最小外接矩形 (Bbox)"""
    if isinstance(segment, ms.Tensor):
        x = segment[..., 0]
        y = segment[..., 1]
    else:
        x, y = segment.T
    return np.array([x.min(), y.min(), x.max(), y.max()])


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算目标边界框惩罚度量 (MindSpore 算子适配版)
    支持 IoU 及 GIoU, DIoU, CIoU 等变体计算
    """
    if xywh:
        x1, y1, w1, h1 = ops.split(box1, 1, axis=-1)
        x2, y2, w2, h2 = ops.split(box2, 1, axis=-1)

        w1_half, h1_half = w1 / 2, h1 / 2
        w2_half, h2_half = w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_half, x1 + w1_half
        b1_y1, b1_y2 = y1 - h1_half, y1 + h1_half
        b2_x1, b2_x2 = x2 - w2_half, x2 + w2_half
        b2_y1, b2_y2 = y2 - h2_half, y2 + h2_half
    else:
        # 分离坐标点并确保各个 Tensor 独立
        b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(box1, 1, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(box2, 1, axis=-1)

    inter = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0) * \
            (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    
    if CIoU or DIoU or GIoU:
        cw = ops.maximum(b1_x2, b2_x2) - ops.minimum(b1_x1, b2_x1)  
        ch = ops.maximum(b1_y2, b2_y2) - ops.minimum(b1_y1, b2_y1)  
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps 
            rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
            if CIoU:
                v = (4 / (math.pi ** 2)) * ops.pow(ops.atan(w2 / h2) - ops.atan(w1 / h1), 2)
                v = ops.stop_gradient(v)
                alpha = v / ((1 - iou) + v + eps)
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
            
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
        
    return iou


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """基于缩放比例系数，将边界框坐标还原回原始图像的空间尺度"""
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  
    boxes[..., [1, 3]] -= pad[1]  
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def dist2bbox(distance, anchor_points, strides=None, xywh=True, axis=-1):
    # 1. 拆分偏移量
    lt, rb = ops.split(distance, split_size_or_sections=2, axis=axis)
        
    # 2. 锚点维度自适应
    ap = anchor_points
    if axis == 1 or axis == -2: # Detect
        if ap.ndim == 2: ap = ops.expand_dims(ap, 0).transpose(0, 2, 1)
        elif ap.ndim == 3 and ap.shape[-1] == 2: ap = ap.transpose(0, 2, 1)
    else: # Segment/Pose
        if ap.ndim == 2: ap = ops.expand_dims(ap, 0)
        elif ap.ndim == 3 and ap.shape[1] == 2: ap = ap.transpose(0, 2, 1)
                
    # 3. 计算坐标
    x1y1 = ap - lt
    x2y2 = ap + rb
        
    # 4. 步长缩放
    if strides is not None:
        s = strides
        if axis == 1 or axis == -2:
            if s.ndim == 1: s = s.view(1, 1, -1)
            elif s.ndim == 2: s = ops.expand_dims(s, 1)
            elif s.ndim == 3 and s.shape[-1] == 1: s = s.transpose(0, 2, 1)
        else:
            if s.ndim == 1: s = s.view(1, -1, 1)
            elif s.ndim == 2: s = ops.expand_dims(s, -1)
            elif s.ndim == 3 and s.shape[1] == 1: s = s.transpose(0, 2, 1)
        x1y1 = x1y1 * s
        x2y2 = x2y2 * s
        
    # 5. 格式输出
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return ops.concat((c_xy, wh), axis=axis)
    return ops.concat((x1y1, x2y2), axis=axis)

def bbox2dist(anchor_points, bbox, reg_max):
    """
    将边界框(x1y1x2y2)转换为相对于锚点的偏移量(ltrb)
    """
    x1y1, x2y2 = ops.split(bbox, split_size_or_sections=2, axis=-1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = ops.concat((lt, rb), axis=-1)
    return dist.clip(0, reg_max - 0.01) # 限制范围防止溢出


def make_divisible(x, divisor):
    """调整特征通道数，使其严格对齐硬件算子的内存对齐规范 (通常为 8 的倍数)"""
    if isinstance(divisor, ms.Tensor):
        divisor = int(divisor.max().asnumpy())
    return math.ceil(x / divisor) * divisor


def batch_iou(batch_box1, batch_box2, eps=1e-7):
    """
    计算批处理数据的交叉 IoU 矩阵
    内置 Batch 截断策略，以兼容 DataLoader 处理尾部不完整批次时的隐式 Padding 行为
    """
    bs1 = batch_box1.shape[0]
    bs2 = batch_box2.shape[0]

    # 应对静态图引擎 Padding 造成的批量不一致情况
    if bs1 != bs2:
        actual_bs = min(bs1, bs2)
        batch_box1 = batch_box1[:actual_bs]
        batch_box2 = batch_box2[:actual_bs]

    b1_x1, b1_y1, b1_x2, b1_y2 = ops.split(batch_box1.expand_dims(2), 1, axis=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = ops.split(batch_box2.expand_dims(1), 1, axis=-1)

    inter_w = (ops.minimum(b1_x2, b2_x2) - ops.maximum(b1_x1, b2_x1)).clip(0)
    inter_h = (ops.minimum(b1_y2, b2_y2) - ops.maximum(b1_y1, b2_y1)).clip(0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = area1 + area2 - inter_area + eps

    return inter_area / union_area


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300, nc=80):
    """
    通用非极大值抑制 (Non-Maximum Suppression, NMS)
    自适应目标检测、实例分割以及姿态估计的特征图切片解析
    
    Args:
        prediction: 网络输出的预测张量
        conf_thres: 用于初筛目标的置信度阈值
        iou_thres: 用于框去重的重叠度阈值
        max_det: 单张图像最大保留的目标数
        nc: 任务的基础类别数，用于动态推断后置的额外特征维度
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
        
    if prediction.shape[1] < prediction.shape[2]:
        prediction = prediction.transpose(0, 2, 1)

    bs = prediction.shape[0]

    # 根据额外的通道特征自动判断是否附带了分割掩码系数或姿态关键点
    extra_dim = prediction.shape[2] - 4 - nc

    output = [ops.zeros((0, 6 + extra_dim), prediction.dtype)] * bs

    for xi, x in enumerate(prediction):
        x = x.astype(ms.float32)

        box = x[:, :4]
        cls = x[:, 4: (4 + nc)]

        conf, cls_id = ops.max(cls, axis=-1)
        mask = conf > conf_thres

        if not mask.any():
            continue

        x_filtered = x[mask]
        curr_box = xywh2xyxy(x_filtered[:, :4])
        curr_conf = conf[mask].expand_dims(-1).astype(ms.float32)
        curr_clsid = cls_id[mask].expand_dims(-1).astype(ms.float32)

        if extra_dim > 0:
            curr_extras = x_filtered[:, (4 + nc): (4 + nc + extra_dim)].astype(ms.float32)
            x_combined = ops.concat((curr_box, curr_conf, curr_clsid, curr_extras), -1)
        else:
            x_combined = ops.concat((curr_box, curr_conf, curr_clsid), -1)

        x_np = x_combined.asnumpy()

        indices = np.argsort(-x_np[:, 4])
        x_np = x_np[indices]

        keep = []
        while x_np.shape[0] > 0:
            current_box = x_np[0]
            keep.append(current_box)
            if x_np.shape[0] == 1:
                break
                
            ious = calculate_nms_iou(current_box[:4], x_np[1:, :4])
            x_np = x_np[1:][ious < iou_thres]

        if len(keep) > 0:
            res = np.stack(keep)
            out_tensor = ms.Tensor(res[:max_det], prediction.dtype)
            output[xi] = out_tensor

    return output


def calculate_nms_iou(box1, boxes):
    """提供纯 NumPy 环境支持的 IoU 计算，辅助完成 NMS"""
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-7)


def xywh2xyxy_np(x):
    """基于 NumPy 的边界框格式解析函数，主要用于脱离计算图的后置评估流"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  
    y[..., 1] = x[..., 1] - x[..., 3] / 2  
    y[..., 2] = x[..., 0] + x[..., 2] / 2  
    y[..., 3] = x[..., 1] + x[..., 3] / 2  
    return y


def crop_mask(masks, boxes):
    """
    掩码边界裁剪操作
    """
    n, h, w = masks.shape

    x1, y1, x2, y2 = ops.split(boxes, 1, axis=-1)
    
    x1 = x1.expand_dims(-1) 
    y1 = y1.expand_dims(-1) 
    x2 = x2.expand_dims(-1) 
    y2 = y2.expand_dims(-1) 

    r_x = ops.arange(w, dtype=ms.float32).view(1, 1, w)
    r_y = ops.arange(h, dtype=ms.float32).view(1, h, 1)

    mask_x = ops.logical_and(r_x >= x1, r_x < x2).astype(ms.float32)
    mask_y = ops.logical_and(r_y >= y1, r_y < y2).astype(ms.float32)
    
    crop_mask_val = mask_x * mask_y

    return masks * crop_mask_val


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    通过模型产生的原型映射层 (Protos) 与掩码系数生成特定目标实例分割图
    支持可选的双线性插值上采样操作
    """
    c, mh, mw = protos.shape
    
    masks_in = masks_in.astype(ms.float32)
    protos_flat = protos.view(c, -1).astype(ms.float32)
    
    masks = ops.matmul(masks_in, protos_flat)
    masks = ops.sigmoid(masks).view(-1, mh, mw)

    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]

    scale_tensor = ms.Tensor([width_ratio, height_ratio, width_ratio, height_ratio], dtype=ms.float32)
    scaled_bboxes = bboxes.astype(ms.float32) * scale_tensor
    
    masks = crop_mask(masks, scaled_bboxes)

    if upsample:
        masks = ops.interpolate(masks.expand_dims(1), size=shape, mode='bilinear', align_corners=False).squeeze(1)

    return ops.cast(masks > 0.5, ms.float32)


def process_mask_native(protos, masks_in, bboxes, shape):
    """原图尺寸下采样后方执行裁剪校验"""
    c, mh, mw = protos.shape
    masks = ops.matmul(masks_in, protos.reshape(c, -1)).reshape(-1, mh, mw)

    masks = ops.interpolate(masks.expand_dims(1), size=shape, mode='bilinear', align_corners=False).squeeze(1)
    masks = crop_mask(masks, bboxes)
    
    return masks > 0


def clip_coords(coords, shape):
    """对点坐标集实行裁剪截断边界约束"""
    h, w = shape[:2]
    if isinstance(coords, ms.Tensor):
        coords[..., 0] = ops.clamp(coords[..., 0], 0, w)
        coords[..., 1] = ops.clamp(coords[..., 1], 0, h)
    else:
        coords[..., 0] = np.clip(coords[..., 0], 0, w)
        coords[..., 1] = np.clip(coords[..., 1], 0, h)
    return coords


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    基于预定义的空间比例，将目标点列 (如姿态点集) 从计算域尺度等距缩放回原图尺度域
    """
    if ratio_pad is None:  
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if isinstance(coords, ms.Tensor):
        coords = coords.astype(ms.float32)
    else:
        coords = np.asarray(coords).astype(np.float32)

    coords[..., 0] -= pad[0]  
    coords[..., 1] -= pad[1]  
    coords[..., 0] /= gain
    coords[..., 1] /= gain

    coords = clip_coords(coords, img0_shape)
    
    return coords