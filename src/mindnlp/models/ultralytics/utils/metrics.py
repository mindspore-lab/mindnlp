import numpy as np
import mindspore as ms
from mindspore import ops

def box_iou(box1, box2, eps=1e-7):
    """
    计算两组边界框的交并比 (IoU)
    
    Args:
        box1: (N, 4) 形状的张量，格式为 [x1, y1, x2, y2]
        box2: (M, 4) 形状的张量，格式为 [x1, y1, x2, y2]
        
    Returns:
        (N, M) 形状的 IoU 矩阵
    """
    be1 = ops.expand_dims(box1, 1)
    be2 = ops.expand_dims(box2, 0)

    lt = ops.maximum(be1[..., :2], be2[..., :2])
    rb = ops.minimum(be1[..., 2:], be2[..., 2:])

    wh = ops.clamp(rb - lt, min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = ops.expand_dims(area1, 1) + ops.expand_dims(area2, 0) - inter + eps

    return inter / union


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    计算各分类的 Average Precision (AP)
    通过对 Precision-Recall 曲线进行数值积分实现
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    ap = np.zeros((nc, tp.shape[1]))
    for ci, c in enumerate(unique_classes):
        i = (pred_cls == c)
        n_l = nt[ci]  
        n_p = i.sum()  

        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc)

        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    return ap, unique_classes


def compute_ap(recall, precision, method='interp'):
    """
    基于 Precision-Recall 序列计算平均精度 (AP)
    支持 101 点插值法 (COCO 默认) 或连续积分法
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # 计算 Precision 包络线，确保单调递减特性
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    if method == 'interp':
        x = np.linspace(0, 1, 101) 
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
    return ap


def process_batch(detections, labels, iou_thresholds):
    """
    基于不同的 IoU 阈值，判定检测结果的真阳性 (TP) 状态
    
    Args:
        detections: (N, 6) 格式为 [x1, y1, x2, y2, conf, cls]
        labels: (M, 5) 格式为 [cls, x1, y1, x2, y2]
        iou_thresholds: 评估用的 IoU 阈值数组
        
    Returns:
        tp: (N, len(iou_thresholds)) 形状的布尔矩阵
    """
    tp = np.zeros((detections.shape[0], iou_thresholds.shape[0]), dtype=bool)

    if labels.shape[0] == 0:  
        return tp

    ious = box_iou(ms.Tensor(detections[:, :4]), ms.Tensor(labels[:, 1:])).asnumpy()

    detected_cls = detections[:, 5]
    target_cls = labels[:, 0]

    for j, iou_thr in enumerate(iou_thresholds):
        matched_gt = np.zeros(labels.shape[0], dtype=bool)

        for i, det in enumerate(detections):
            cls = det[5]
            matches = np.where((target_cls == cls) & (matched_gt == 0) & (ious[i] >= iou_thr))[0]

            if matches.shape[0] > 0:
                best_match = matches[ious[i, matches].argmax()]
                tp[i, j] = True
                matched_gt[best_match] = True  

    return tp


def mask_iou(mask1, mask2, eps=1e-7):
    """计算像素级掩码的交并比 (Mask IoU)"""
    m1 = mask1.view(mask1.shape[0], -1).astype(ms.float32)
    m2 = mask2.view(mask2.shape[0], -1).astype(ms.float32)

    intersection = ops.matmul(m1, m2.T)
    area1 = m1.sum(1).expand_dims(1)
    area2 = m2.sum(1).expand_dims(0)
    union = area1 + area2 - intersection + eps

    return intersection / union


# 各类任务评估计算

class ClassifyMetrics:
    """图像分类任务评估类，计算 Top-1 和 Top-5 准确率"""
    def __init__(self):
        self.top1 = 0.0
        self.top5 = 0.0
        self.task = "classify"

    def process(self, targets, preds):
        if not targets or not preds:
            return

        preds = np.concatenate(preds, axis=0)      
        targets = np.concatenate(targets, axis=0)  

        correct = (targets[:, None] == preds)      

        self.top1 = float(correct[:, 0].mean())
        self.top5 = float(correct.any(axis=1).mean())

    @property
    def fitness(self):
        """分类任务适应度得分 (Top-1 与 Top-5 的均值)"""
        return (self.top1 + self.top5) / 2.0

    @property
    def results_dict(self):
        return {
            "metrics/accuracy_top1": self.top1,
            "metrics/accuracy_top5": self.top5,
            "fitness": self.fitness
        }


class DetMetrics:
    """目标检测任务评估类，维护预测与标签状态，计算全类别 mAP"""
    def __init__(self, names):
        self.names = names
        self.nc = len(names)
        self.stats = []

    def update_stats(self, tp, conf, pred_cls, target_cls):
        self.stats.append({
            "tp": tp, "conf": conf, "pred_cls": pred_cls, "target_cls": target_cls
        })

    def process(self):
        if not self.stats:
            return {"metrics/mAP50(B)": 0, "metrics/mAP50-95(B)": 0}

        tp = np.concatenate([x["tp"] for x in self.stats], 0)
        conf = np.concatenate([x["conf"] for x in self.stats], 0)
        pred_cls = np.concatenate([x["pred_cls"] for x in self.stats], 0)
        target_cls = np.concatenate([x["target_cls"] for x in self.stats], 0)

        ap, _ = ap_per_class(tp, conf, pred_cls, target_cls)

        return {
            "metrics/mAP50(B)": ap[:, 0].mean(),
            "metrics/mAP50-95(B)": ap.mean()
        }

    @property
    def fitness(self):
        """检测任务适应度得分：0.1 * mAP@50 + 0.9 * mAP@50-95"""
        stats = self.process()
        return stats.get("metrics/mAP50(B)", 0) * 0.1 + stats.get("metrics/mAP50-95(B)", 0) * 0.9


class SegmentMetrics(DetMetrics):
    """实例分割任务评估类，计算边界框及掩码的双重 mAP"""
    def __init__(self, names):
        super().__init__(names)
        self.stats_m = []

    def update_stats(self, tp_b, tp_m, conf, pred_cls, target_cls):
        super().update_stats(tp_b, conf, pred_cls, target_cls)
        self.stats_m.append(tp_m)

    def process(self):
        results = super().process()
        if not self.stats_m:
            return results

        tp_m = np.concatenate(self.stats_m, 0)
        ap_m, _ = ap_per_class(
            tp_m,
            np.concatenate([x["conf"] for x in self.stats], 0),
            np.concatenate([x["pred_cls"] for x in self.stats], 0),
            np.concatenate([x["target_cls"] for x in self.stats], 0)
        )

        results.update({
            "metrics/mAP50(M)": ap_m[:, 0].mean(),
            "metrics/mAP50-95(M)": ap_m.mean()
        })
        return results

    @property
    def fitness(self):
        stats = self.process()
        # 计算目标检测 (Box) 的分数
        box_fitness = stats.get("metrics/mAP50(B)", 0) * 0.1 + stats.get("metrics/mAP50-95(B)", 0) * 0.9
        # 计算实例分割 (Mask) 的分数
        mask_fitness = stats.get("metrics/mAP50(M)", 0) * 0.1 + stats.get("metrics/mAP50-95(M)", 0) * 0.9
        # 综合两者的表现
        return box_fitness + mask_fitness


OKS_SIGMA = np.array(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89],
    dtype=np.float32,
) / 10.0


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """计算关键点目标相似度 (Object Keypoint Similarity, OKS)"""
    p_xy = ops.expand_dims(kpt1[..., :2], 1)  
    g_xy = ops.expand_dims(kpt2[..., :2], 0)  

    d2 = ops.pow(p_xy - g_xy, 2).sum(-1)      

    g_mask = kpt2[..., 2] > 0                 
    g_mask_exp = ops.expand_dims(g_mask, 0)   
    
    sigma = ms.Tensor(sigma, dtype=kpt1.dtype)
    sig2 = ops.pow(sigma * 2, 2)              
    area_exp = area.view(1, -1, 1)            
    
    denominator = area_exp * sig2 * 2         

    exponent = -d2 / (denominator + eps)
    exponent = ops.clamp(exponent, -100, 0)
    oks_all = ops.exp(exponent)

    g_mask_float = g_mask_exp.astype(kpt1.dtype)
    num = (oks_all * g_mask_float).sum(-1)       
    den = g_mask_float.sum(-1) + eps             

    return num / den


class PoseMetrics(DetMetrics):
    """姿态估计任务评估类，计算边界框及关键点 OKS 的双重 mAP"""
    def __init__(self, names):
        super().__init__(names)
        self.stats_p = []

    def update_stats(self, tp_b, tp_p, conf, pred_cls, target_cls):
        super().update_stats(tp_b, conf, pred_cls, target_cls)
        self.stats_p.append(tp_p)

    def process(self):
        results = super().process()
        if not self.stats_p:
            return results

        tp_p = np.concatenate(self.stats_p, 0)
        ap_p, _ = ap_per_class(
            tp_p,
            np.concatenate([x["conf"] for x in self.stats], 0),
            np.concatenate([x["pred_cls"] for x in self.stats], 0),
            np.concatenate([x["target_cls"] for x in self.stats], 0)
        )

        results.update({
            "metrics/mAP50(P)": ap_p[:, 0].mean() if len(ap_p) else 0.0,
            "metrics/mAP50-95(P)": ap_p.mean() if len(ap_p) else 0.0
        })
        return results

    @property
    def fitness(self):
        stats = self.process()
        # 计算目标检测 (Box) 的分数
        box_fitness = stats.get("metrics/mAP50(B)", 0) * 0.1 + stats.get("metrics/mAP50-95(B)", 0) * 0.9
        # 计算姿态估计 (Pose/Keypoint) 的分数
        pose_fitness = stats.get("metrics/mAP50(P)", 0) * 0.1 + stats.get("metrics/mAP50-95(P)", 0) * 0.9
        # 综合两者的表现
        return box_fitness + pose_fitness