import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor

from utils.tal import TaskAlignedAssigner, make_anchors
from utils.ops import bbox_iou, xywh2xyxy, dist2bbox, bbox2dist

# COCO 数据集 17 个关键点的标准差（用于计算 OKS 相似度）
OKS_SIGMA = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


# 分类任务损失函数
class YOLOClassificationLoss(nn.Cell):
    """YOLO11 图像分类任务损失函数 (基于交叉熵)"""
    def __init__(self):
        super(YOLOClassificationLoss, self).__init__()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, preds, targets):
        if isinstance(targets, dict):
            label_tensor = targets.get('label', targets.get('cls')) 
        else:
            label_tensor = targets
            
        return self.loss_fn(preds, label_tensor)

class v8ClassificationLoss(nn.Cell):
    """YOLO 多标签分类逻辑损失函数 (基于 BCE)"""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def construct(self, preds, targets): 
        num_classes = preds.shape[-1]
        dtype = preds.dtype
        on_value = Tensor(1.0, dtype)
        off_value = Tensor(0.0, dtype)
        
        # 修复：展平 targets 防止多出维度
        targets_flat = targets.view(-1) 
        targets_one_hot = ops.one_hot(targets_flat, num_classes, on_value, off_value)
        
        return self.loss_fn(preds, targets_one_hot)

# 目标检测基础损失组件

class DFLoss(nn.Cell):
    """分布焦点损失 (Distribution Focal Loss)"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def construct(self, pred_dist, target):
        # 将输入分布重塑为交叉熵格式 [N * 4, reg_max]
        pred_dist = pred_dist.view(-1, self.reg_max)

        # 拆分目标边界偏移量以计算积分权重
        target_left = ops.cast(target, ms.int32)
        target_right = target_left + 1

        weight_left = target_right.astype(pred_dist.dtype) - target
        weight_right = target - target_left.astype(pred_dist.dtype)

        tl = target_left.view(-1)
        tr = target_right.view(-1)

        loss_left = ops.cross_entropy(pred_dist, tl, reduction="none").view(target.shape) * weight_left
        loss_right = ops.cross_entropy(pred_dist, tr, reduction="none").view(target.shape) * weight_right

        return (loss_left + loss_right).mean()


class BboxLoss(nn.Cell):
    """边界框回归损失 (包含 CIoU/GIoU 损失与 DFL 损失)"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def construct(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        fg_indices = ops.nonzero(fg_mask)
        loss_iou = ops.zeros((), ms.float32)
        loss_dfl = ops.zeros((), ms.float32)

        if fg_indices.shape[0] > 0:
            p_boxes = ops.gather_nd(pred_bboxes, fg_indices).astype(ms.float32)
            t_boxes = ops.gather_nd(target_bboxes, fg_indices).astype(ms.float32)
            weight = ops.gather_nd(target_scores.sum(-1), fg_indices).expand_dims(-1).astype(ms.float32)

            # 计算 IoU 损失并限制范围以确保数值稳定性
            iou = bbox_iou(p_boxes, t_boxes, xywh=False, CIoU=False).clip(0, 1)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

            # 计算 DFL 损失
            if self.dfl_loss:
                selected_p_dist = ops.gather_nd(pred_dist, fg_indices)
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                selected_t_ltrb = ops.gather_nd(target_ltrb, fg_indices)
                loss_dfl = self.dfl_loss(selected_p_dist, selected_t_ltrb)

        return loss_iou, loss_dfl


# YOLO11 主干任务损失类

class v8DetectionLoss(nn.Cell):
    """YOLO11 目标检测任务损失函数计算类"""
    def __init__(self, model):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.nc = model.nc
        self.imgsz = getattr(model, 'imgsz', 640)
        self.reg_max = model.reg_max
        self.bbox_loss = BboxLoss(self.reg_max)
        self.stride = model.stride

        # 动态分配器
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = ops.arange(self.reg_max).astype(ms.float32)
        
        # 将默认超参数转化为类属性，避免硬编码
        self.hyp_box = getattr(model, 'hyp_box', 7.5)
        self.hyp_cls = getattr(model, 'hyp_cls', 0.5)
        self.hyp_dfl = getattr(model, 'hyp_dfl', 1.5)

    def decode_predictions(self, feats):
        """
        解析特征图，分离分类得分与边界框分布
        
        Args:
            feats (Tensor): [Batch, 4*reg_max + nc, Anchors] 形状的展平张量
            
        Returns:
            pred_distri (Tensor): 边界框分布信息
            pred_scores (Tensor): 分类得分
        """
        # 如果传入的是未拼接的原始特征图列表，执行自动拼接 (兼容性逻辑)
        if isinstance(feats, (list, tuple)):
            x_list = []
            for xi in feats:
                b, c, h, w = xi.shape
                x_list.append(xi.view(b, c, -1))
            x = ops.concat(x_list, axis=2)
        else:
            x = feats

        # 按通道分割分类与回归信息
        pred_distri, pred_scores = ops.split(x, (self.reg_max * 4, self.nc), axis=1)
        
        # 交换轴以适配后续计算：[B, C, N] -> [B, N, C]
        return pred_distri.swapaxes(1, 2), pred_scores.swapaxes(1, 2)

    def bbox_decode(self, anchor_points, pred_dist):
        """
        通过积分机制将预测分布解码为物理边界框坐标
        
        Args:
            anchor_points (Tensor): 锚点中心坐标
            pred_dist (Tensor): 预测的 ltrb 分布
        """
        if self.reg_max > 1:
            batch, anchors, channels = pred_dist.shape

            pred_dist = pred_dist.view(batch, anchors, 4, self.reg_max)
            pred_dist = ops.softmax(pred_dist, axis=-1)
            pred_dist = ops.matmul(pred_dist, self.proj)

        return dist2bbox(pred_dist, anchor_points[:pred_dist.shape[1]], xywh=False)

    def preprocess_targets(self, batch, feats):
        """
        将散装标签 (Collapsed Labels) 重组为标准 Batch 格式
        解决 [Batch_Idx, Class, x, y, w, h] 格式的维度对齐问题
        """
        input_tensor = feats[0] if isinstance(feats, (list, tuple)) else feats
        bs = input_tensor.shape[0]
        
        bboxes = batch["bboxes"]
        cls = batch["cls"]
        batch_idx = batch["batch_idx"].view(-1)
        
        if not isinstance(bboxes, ms.Tensor):
            bboxes = ms.Tensor(bboxes, ms.float32)
        if not isinstance(cls, ms.Tensor):
            cls = ms.Tensor(cls, ms.float32)

        # 统计每张图的目标数并确定本批次最大目标量，用于 Tensor 填充
        counts = []
        for i in range(bs):
            counts.append(int(ops.sum(batch_idx == i).asnumpy()))
        max_obj = max(max(counts), 1)

        # 构建标准的 [Batch, Max_Obj, 5] 容器
        targets = ops.zeros((bs, max_obj, 5), bboxes.dtype)

        for i in range(bs):
            mask = (batch_idx == i)
            num = counts[i]
            if num > 0:
                actual_num = min(num, max_obj)
                # 填充 Class ID 与 Bbox 坐标
                targets[i, :actual_num, 0] = cls[mask][:actual_num].view(-1)
                targets[i, :actual_num, 1:5] = bboxes[mask][:actual_num]

        # 坐标映射：将归一化坐标转换为像素坐标
        if hasattr(self, 'imgsz') and self.imgsz:
            targets[..., 1:5] *= self.imgsz
            
        return targets

    def construct(self, preds, batch):
        """
        Loss 计算图主入口
        """
        # 解包模型输出：x_cat 为展平预测值，feats_list 为多尺度特征列表
        if isinstance(preds, (tuple, list)):
            x_cat = preds[0]
            feats_list = preds[1]
        else:
            x_cat = preds
            feats_list = [preds]

        # 解析预测分布与分类得分
        pred_distri, pred_scores = self.decode_predictions(x_cat)
        
        # 生成锚点中心与步长向量 (基于特征图原始分辨率)
        anchor_points, stride_tensor = make_anchors(feats_list, self.stride, 0.5)

        # 标签重组与拆分
        targets = self.preprocess_targets(batch, feats_list)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # [B, N, 1], [B, N, 4]
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0    # 有效目标掩码

        # 坐标解码：生成解码后的预测框并转换真实框格式
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        current_gt_bboxes = xywh2xyxy(gt_bboxes)

        # 正负样本分配 (Assigner)
        # 注意：此处输入均已还原至像素空间
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.sigmoid(),
            (pred_bboxes * stride_tensor),
            anchor_points * stride_tensor,
            gt_labels,
            current_gt_bboxes,
            mask_gt
        )[1:4]

        # 损失计算逻辑
        target_scores_sum = ops.maximum(target_scores.sum(), 1.0)
        
        # 分类损失 (BCE)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        # 回归损失 (IoU + DFL)
        loss_box, loss_dfl = ops.zeros((), ms.float32), ops.zeros((), ms.float32)
        fg_mask_bool = ops.cast(fg_mask, ms.bool_)

        if fg_mask_bool.any():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask_bool
            )

        # 记录 Loss Items 用于日志展示 (stop_gradient 保证记录不干扰反传)
        self.loss_items = ops.stop_gradient(ops.stack([
            loss_box * self.hyp_box, 
            loss_cls * self.hyp_cls, 
            loss_dfl * self.hyp_dfl
        ]))
        
        return (loss_box * self.hyp_box), (loss_cls * self.hyp_cls), (loss_dfl * self.hyp_dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """YOLO11 实例分割任务损失函数计算类"""
    def __init__(self, model):
        super().__init__(model)
        self.overlap = True
        self.hyp_seg = getattr(model, 'hyp_seg', 2.5)

    def preprocess_masks(self, batch, mask_h, mask_w, bs):
        """将扁平化的分割掩码打包为批量张量"""
        masks = batch["masks"].astype(ms.float32)
        batch_idx = batch["batch_idx"].view(-1)

        if masks.shape[0] == 0:
            return ops.zeros((bs, 1, mask_h, mask_w), ms.float32)

        if masks.shape[-2:] != (mask_h, mask_w):
            masks = ops.ResizeNearestNeighbor((mask_h, mask_w))(masks.expand_dims(1))[:, 0]

        counts = [int((batch_idx == i).sum()) for i in range(bs)]
        max_obj = max(max(counts) if counts else 0, 1)

        target_masks = ops.zeros((bs, max_obj, mask_h, mask_w), ms.float32)
        for i in range(bs):
            mask_flag = (batch_idx == i)
            num = counts[i]
            if num > 0:
                actual_num = min(num, max_obj)
                target_masks[i, :actual_num] = masks[mask_flag][:actual_num]

        return target_masks

    def calculate_segmentation_loss(self, fg_mask, gt_masks, target_gt_idx, target_bboxes, proto, pred_masks):
        """计算掩码交叉熵损失"""
        bs, nm, ph, pw = proto.shape
        loss_mask = ops.zeros((), ms.float32)

        for i in range(bs):
            mask_indices = ops.nonzero(fg_mask[i]).view(-1)
            if mask_indices.shape[0] == 0:
                continue

            coeffs = pred_masks[i][mask_indices]
            p = proto[i].view(nm, -1)
            pred_mask_full = ops.matmul(coeffs, p).view(-1, ph, pw)

            current_target_idx = target_gt_idx[i].view(-1)
            obj_idx = ops.cast(current_target_idx[mask_indices], ms.int32)
            target_mask = ops.cast(gt_masks[i][obj_idx] > 0, ms.float32)

            t_box = target_bboxes[i][mask_indices]
            ratio_w = pw / getattr(self, 'imgsz', 640)
            ratio_h = ph / getattr(self, 'imgsz', 640)

            x1 = (t_box[:, 0] * ratio_w).view(-1, 1, 1)
            y1 = (t_box[:, 1] * ratio_h).view(-1, 1, 1)
            x2 = (t_box[:, 2] * ratio_w).view(-1, 1, 1)
            y2 = (t_box[:, 3] * ratio_h).view(-1, 1, 1)

            r_x = ops.arange(pw).view(1, 1, pw).astype(ms.float32)
            r_y = ops.arange(ph).view(1, ph, 1).astype(ms.float32)

            crop_mask = ops.cast((r_x >= x1) & (r_x < x2) & (r_y >= y1) & (r_y < y2), ms.float32)
            target_mask = target_mask * crop_mask

            loss_m = ops.binary_cross_entropy_with_logits(pred_mask_full, target_mask, reduction="none")
            loss_mask += loss_m.mean(axis=(1, 2)).sum()

        return loss_mask

    def construct(self, preds, batch):
        feats, pred_masks, proto = preds
        bs, nm, mask_h, mask_w = proto.shape

        pred_distri, pred_scores = self.decode_predictions(feats)
        pred_masks = pred_masks.transpose(0, 2, 1)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        targets = self.preprocess_targets(batch, feats)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)

        if gt_bboxes.max() <= 1.0:
            gt_bboxes = gt_bboxes * getattr(self, 'imgsz', 640)
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        current_gt_bboxes = xywh2xyxy(gt_bboxes)

        pred_scores_sg = ops.stop_gradient(pred_scores.sigmoid())
        pred_bboxes_sg = ops.stop_gradient(pred_bboxes * stride_tensor)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores_sg, pred_bboxes_sg, anchor_points * stride_tensor,
            gt_labels, current_gt_bboxes, mask_gt
        )

        target_scores_sum = ops.maximum(target_scores.sum(), 1.0)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum

        loss_box, loss_dfl, loss_mask = ops.zeros((), ms.float32), ops.zeros((), ms.float32), ops.zeros((), ms.float32)

        if fg_mask.any():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask
            )
            gt_masks = self.preprocess_masks(batch, mask_h, mask_w, bs)
            loss_mask = self.calculate_segmentation_loss(
                fg_mask, gt_masks, target_gt_idx, target_bboxes, proto, pred_masks
            )
            loss_mask = loss_mask / target_scores_sum

        return (loss_box * self.hyp_box), (loss_cls * self.hyp_cls), (loss_dfl * self.hyp_dfl), (loss_mask * self.hyp_seg)


class KeypointLoss(nn.Cell):
    """关键点估计损失计算 (基于目标关键点相似度 OKS)"""
    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = ms.Tensor(sigmas, ms.float32)

    def construct(self, pred_kpts, gt_kpts, kpt_mask, area):
        pred_kpts = ops.cast(pred_kpts, ms.float32)
        gt_kpts = ops.cast(gt_kpts, ms.float32)
        kpt_mask_f = ops.cast(kpt_mask, ms.float32)
        area = ops.cast(area, ms.float32)

        d = ops.pow(pred_kpts[..., 0] - gt_kpts[..., 0], 2) + ops.pow(pred_kpts[..., 1] - gt_kpts[..., 1], 2)
        factor = pred_kpts.shape[1] / (ops.sum(kpt_mask_f, dim=1) + 1e-9)
        sigmas_sq = ops.pow(2 * self.sigmas, 2)
        
        e = d / (sigmas_sq * (area + 1e-9) * 2)
        loss = factor.expand_dims(-1) * ((1.0 - ops.exp(-e)) * kpt_mask_f)
        
        return loss.mean(axis=1).sum()


class v8PoseLoss(v8DetectionLoss):
    """YOLO11 姿态估计任务专用损失函数计算类"""
    def __init__(self, model):
        super().__init__(model)
        self.kpt_shape = getattr(model.model[-1], 'kpt_shape', [17, 3])
        self.nkpt = self.kpt_shape[0]
        self.ndim = self.kpt_shape[1]
        self.bce_pose = nn.BCEWithLogitsLoss(reduction="mean")
        
        sigmas = OKS_SIGMA if self.kpt_shape == [17, 3] else (np.ones(self.nkpt) / self.nkpt)
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        
        self.hyp_pose = getattr(model, 'hyp_pose', 12.0)
        self.hyp_kobj = getattr(model, 'hyp_kobj', 1.0)

    def preprocess_keypoints(self, batch, bs):
        """格式化数据集目标关键点"""
        keypoints = batch["keypoints"].astype(ms.float32)
        batch_idx = batch["batch_idx"].view(-1)
        counts = [int((batch_idx == i).sum()) for i in range(bs)]
        max_obj = max(max(counts) if counts else 0, 1)
        
        targets_kpt = ops.zeros((bs, max_obj, self.nkpt, self.ndim), keypoints.dtype)
        for i in range(bs):
            num = counts[i]
            if num > 0:
                actual_num = min(num, max_obj)
                targets_kpt[i, :actual_num] = keypoints[batch_idx == i][:actual_num]

        if hasattr(self, 'imgsz') and self.imgsz:
            targets_kpt[..., 0] *= self.imgsz
            targets_kpt[..., 1] *= self.imgsz

        return targets_kpt

    def kpts_decode(self, anchor_points, pred_kpts, stride_tensor):
        """还原预测关键点的物理坐标"""
        anc = anchor_points.view(1, anchor_points.shape[0], 1, 2)
        strides = stride_tensor.view(1, anchor_points.shape[0], 1, 1) 
        xy = (pred_kpts[..., :2] * 2.0 + (anc - 0.5)) * strides
        vis = pred_kpts[..., 2:3]
        return ops.concat((xy, vis), axis=-1)

    def calculate_keypoints_loss(self, fg_mask, target_gt_idx, targets_kpt, stride_tensor, target_bboxes, pred_kpts_decoded, pred_kpts_raw):
        bs = fg_mask.shape[0]
        loss_pose, loss_kobj = ops.zeros((), ms.float32), ops.zeros((), ms.float32)

        for i in range(bs):
            mask_indices = ops.nonzero(fg_mask[i]).view(-1)
            if mask_indices.shape[0] == 0:
                continue

            p_kpt_dec = pred_kpts_decoded[i][mask_indices]  
            p_kpt_raw = pred_kpts_raw[i][mask_indices]      
            current_target_idx = target_gt_idx[i].reshape(-1)
            obj_idx = current_target_idx[mask_indices]
            gt_kpt = targets_kpt[i][obj_idx]  
            
            gt_kpt_pixel = gt_kpt[..., :2] 
            t_box = target_bboxes[i][mask_indices]
            
            width_f32 = ops.cast(t_box[:, 2] - t_box[:, 0], ms.float32)
            height_f32 = ops.cast(t_box[:, 3] - t_box[:, 1], ms.float32)
            area_pixel = (width_f32 * height_f32).view(-1, 1)

            kpt_mask = gt_kpt[..., 2] > 0
            loss_pose += self.keypoint_loss(p_kpt_dec[..., :2], gt_kpt_pixel, kpt_mask, area_pixel)

            if self.ndim == 3:
                kpt_mask_f = ops.cast(kpt_mask, ms.float32)
                loss_kobj += self.bce_pose(p_kpt_raw[..., 2], kpt_mask_f)

        return loss_pose, loss_kobj

    def construct(self, preds, batch):
        feats, pred_kpts = preds[0], preds[1]
        pred_distri, pred_scores = self.decode_predictions(feats)
        bs = pred_scores.shape[0]

        kpt_list = [pk.view(pk.shape[0], pk.shape[1], -1) for pk in pred_kpts]
        pred_kpts_cat = ops.concat(kpt_list, axis=2)
        pred_kpts = pred_kpts_cat.swapaxes(1, 2).view(bs, -1, self.nkpt, self.ndim)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = self.preprocess_targets(batch, feats)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        
        if gt_bboxes.max() <= 1.0:
            gt_bboxes = gt_bboxes * getattr(self, 'imgsz', 640)
            
        mask_gt = gt_bboxes.sum(2, keepdim=True) > 0
        targets_kpt = self.preprocess_keypoints(batch, bs)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        current_gt_bboxes = xywh2xyxy(gt_bboxes)
        
        pred_scores_sg = ops.stop_gradient(pred_scores.sigmoid())
        pred_bboxes_sg = ops.stop_gradient(pred_bboxes * stride_tensor)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores_sg, pred_bboxes_sg, anchor_points * stride_tensor,
            gt_labels, current_gt_bboxes, mask_gt
        )

        fg_mask_bool = ops.cast(fg_mask, ms.bool_)
        pred_kpts_decoded = self.kpts_decode(anchor_points, pred_kpts, stride_tensor)
        
        target_scores_sum = ops.maximum(target_scores.sum(), 1.0)
        loss_cls = self.bce(pred_scores, target_scores).sum() / target_scores_sum
        
        loss_box, loss_dfl, loss_pose, loss_kobj = ops.zeros((), ms.float32), ops.zeros((), ms.float32), ops.zeros((), ms.float32), ops.zeros((), ms.float32)

        if fg_mask_bool.any():
            loss_box, loss_dfl = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes / stride_tensor, target_scores, target_scores_sum, fg_mask_bool
            )
            loss_pose, loss_kobj = self.calculate_keypoints_loss(
                fg_mask_bool, target_gt_idx, targets_kpt,
                stride_tensor, target_bboxes, pred_kpts_decoded, pred_kpts
            )
            loss_pose /= target_scores_sum

        return (loss_box * self.hyp_box), (loss_cls * self.hyp_cls), (loss_dfl * self.hyp_dfl), (loss_pose * self.hyp_pose), (loss_kobj * self.hyp_kobj)