import numpy as np
import mindspore as ms
from mindspore import nn, ops

# 规范引入算子
from utils.ops import batch_iou

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """
    根据特征图的尺度生成对应的锚点中心坐标与步长张量
    
    Args:
        feats (list[Tensor]): 多尺度特征图列表
        strides (list[int]): 对应特征图的下采样步长
        grid_cell_offset (float): 网格中心点偏移量，默认 0.5
        
    Returns:
        tuple[Tensor, Tensor]: 展平后的锚点坐标张量与对应的步长张量
    """
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype

    for i, stride in enumerate(strides):
        if i >= len(feats):
            break

        shape = feats[i].shape
        # 解析特征图空间维度 (支持 [B, C, H, W] 或 [B, H, W] 格式)
        if len(shape) == 4:
            _, _, h, w = shape
        elif len(shape) == 3:  
            _, h, w = shape
        else:
            raise ValueError(f"[ERROR] 特征图维度解析失败，期望维度为 3 或 4，实际为 {len(shape)}。")

        sx = ops.arange(w).astype(dtype) + grid_cell_offset
        sy = ops.arange(h).astype(dtype) + grid_cell_offset

        # 生成网格坐标 (基于 ij 索引)
        grid_y, grid_x = ops.meshgrid(sy, sx, indexing='ij')

        anchor_points.append(ops.stack((grid_x, grid_y), -1).view(-1, 2))
        stride_tensor.append(ops.full((h * w, 1), stride, dtype=dtype))

    return ops.concat(anchor_points, 0), ops.concat(stride_tensor, 0)


class TaskAlignedAssigner(nn.Cell):
    """
    任务对齐样本分配器 (Task-Aligned Assigner)
    依据分类得分与边界框 IoU 的加权指标，为真实目标动态分配正样本锚点
    """
    def __init__(self, topk=10, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def construct(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        任务对齐分配器：修正冲突消解与软标签逻辑。
        """
        bs = pd_bboxes.shape[0]
        num_anchors = pd_bboxes.shape[1]
        n_max_boxes = gt_bboxes.shape[1]

        # 1. 计算 IoU [bs, n_max_boxes, num_anchors]
        ious = batch_iou(gt_bboxes, pd_bboxes)
        if ious.ndim == 4: ious = ious.squeeze(-1)

        # 2. 提取对应 GT 类别的预测得分
        t_labels = gt_labels.squeeze(-1).astype(ms.int32)
        # 构造索引矩阵 [bs, n_max_boxes, num_anchors]
        scores = ops.gather_elements(
            pd_scores.transpose(0, 2, 1),
            1,
            ops.broadcast_to(t_labels.expand_dims(-1), (bs, n_max_boxes, num_anchors))
        )

        # 3. 计算对齐度 Alignment Metric
        align_metric = ops.pow(scores, self.alpha) * ops.pow(ious, self.beta)

        # 4. 选取 Top-k 候选
        topk_values, topk_indices = ops.topk(align_metric, self.topk, dim=-1)
        is_in_topk = ops.tensor_scatter_elements(
            ops.zeros_like(align_metric), topk_indices, ops.ones_like(topk_values), axis=2
        )

        # 5. 中心点过滤 (确保锚点在 GT 框内)
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        mask_pos = is_in_topk * mask_in_gts.astype(ms.float32) * mask_gt.astype(ms.float32)

        # 6. 每个锚点只允许对应一个最优 GT
        mask_pos_sum = mask_pos.sum(-2) # [bs, num_anchors]
        if (mask_pos_sum > 1).any():
            # 找到每个锚点 metric 最大的 GT 索引
            max_idx = align_metric.argmax(1) # [bs, num_anchors]
            one_hot_mask = ops.one_hot(max_idx, n_max_boxes, 1.0, 0.0).transpose(0, 2, 1)
            mask_pos *= one_hot_mask

        # 7. 生成最终标签
        fg_mask = mask_pos.sum(1) > 0 # 前景掩码
        best_gt_idx = mask_pos.argmax(1) # [bs, num_anchors]

        # 提取 target_bboxes
        target_bboxes = ops.gather_elements(
            ops.broadcast_to(gt_bboxes.expand_dims(1), (bs, num_anchors, n_max_boxes, 4)),
            2,
            best_gt_idx.view(bs, num_anchors, 1, 1).expand_as(ops.zeros((bs, num_anchors, 1, 4), ms.int32))
        ).squeeze(2)

        # 提取 target_labels 并生成 Soft Targets (分类分 * IoU)
        target_labels = ops.gather_elements(t_labels, 1, best_gt_idx)
        target_scores = ops.one_hot(target_labels, self.num_classes, 1.0, 0.0)
        
        # 结合 IoU 调整置信度
        max_iou = ops.gather_elements(ious.transpose(0, 2, 1), 2, best_gt_idx.expand_dims(-1)).squeeze(-1)
        target_scores *= max_iou.expand_dims(-1)
        target_scores *= fg_mask.expand_dims(-1).astype(ms.float32)

        return target_labels, target_bboxes, target_scores, fg_mask.astype(ms.bool_), best_gt_idx

    def select_candidates_in_gts(self, anc_points, gt_bboxes):
        """
        中心点空间过滤策略
        判定网格中心锚点是否物理落入对应的真实边界框范围内
        """
        anc_points = anc_points.expand_dims(0).expand_dims(0)

        lt = gt_bboxes[..., :2].expand_dims(2)  
        rb = gt_bboxes[..., 2:].expand_dims(2)  

        # 计算从中心点到边界框四个边的几何增量
        bbox_deltas = ops.concat((anc_points - lt, rb - anc_points), axis=-1)

        # 空间范围校验：所有几何增量必须大于极小值
        return bbox_deltas.amin(axis=-1) > self.eps
