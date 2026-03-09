import logging
import numpy as np
import cv2
import mindspore as ms
from mindspore import Tensor, ops

from engine.validator import BaseValidator
from utils.ops import non_max_suppression, process_mask, xywh2xyxy_np
from utils.metrics import SegmentMetrics

LOGGER = logging.getLogger(__name__)

class SegmentationValidator(BaseValidator):
    """
    实例分割任务评估验证器
    重写 BaseValidator 的前处理及后处理节点，以计算 BBox mAP 与 Mask mAP 双重验证指标
    """
    def __init__(self, dataloader=None, save_dir=None, args=None, names=None):
        super().__init__(dataloader, save_dir, args)
        self.names = names
        self.metrics = None
        self.stats = {}
        self.niou = 10  
        self.iou_v = np.linspace(0.5, 0.95, self.niou)

    def get_dataloader(self, dataset_path, batch_size=16):
        """激活具备 Mask 特征字段打包逻辑的数据加载器"""
        return create_dataloader(
            path=dataset_path,
            imgsz=getattr(self.args, 'imgsz', 640),
            batch_size=batch_size,
            task='segment',  
            is_training=False,
            num_workers=getattr(self.args, 'workers', 8)
        )

    def init_metrics(self, model):
        """建立分工独立的 BBox 与 Mask 精度统计容器"""
        if not self.names:
            self.names = getattr(model, 'names', {i: f'class_{i}' for i in range(getattr(self.args, 'nc', 80))})
        self.metrics = SegmentMetrics(names=self.names)

    def preprocess(self, batch):
        """
        预处理阶段：确保图像输入浮点化域为 [0, 1]
        统一目标真值边框 (GT BBoxes) 的数据格式为物理像素下的绝对坐标 (xyxy)
        """
        batch["image"] = ops.cast(batch["image"], ms.float32) / 255.0
        
        _, _, h, w = batch["image"].shape 
        bboxes_np = batch["bboxes"].asnumpy()
        
        if len(bboxes_np) > 0:
            bboxes_xyxy = xywh2xyxy_np(bboxes_np)
            
            # 若原始数据停留在归一化相对空间，则实施放大操作
            if bboxes_xyxy.max() <= 1.01:
                bboxes_xyxy[:, [0, 2]] *= w
                bboxes_xyxy[:, [1, 3]] *= h
                
            batch["bboxes"] = ms.Tensor(bboxes_xyxy, dtype=ms.float32)
            
        return batch

    def postprocess(self, preds):
        """
        后处理逻辑链：从网络提取原生预测簇，利用 NMS 算子进行空间重叠度消解
        验证阶段要求捕获极大限度的召回率边界，故应用预设的超低置信度阈值
        """
        preds_box, proto = preds[0], preds[1]
        
        # 为了捕获完整的 Precision-Recall 曲线全貌，验证期的 conf_thres 被极小化约束
        iou_thres = self.hyp.get('iou', 0.6)
        preds_nms = non_max_suppression(
            preds_box, 
            conf_thres=0.001,  
            iou_thres=iou_thres,
            max_det=300
        )
        return preds_nms, proto

    def update_metrics(self, preds, batch):
        """计算图像批次内的真实阳性实例 (TP)，记录针对 Mask 与 Box 的双模评估指标"""
        preds_nms, proto = preds
        
        batch_idx = batch["batch_idx"].view(-1).asnumpy()
        all_gt_cls = batch["cls"].view(-1).asnumpy()
        all_gt_bboxes = batch["bboxes"].asnumpy()
        all_gt_masks = batch["masks"]
        
        imgsz = getattr(self.args, 'imgsz', 640)

        for i, pred in enumerate(preds_nms):
            mask_gt = (batch_idx == i)
            gt_cls = all_gt_cls[mask_gt]
            gt_bboxes = all_gt_bboxes[mask_gt] 
            gt_masks = all_gt_masks[mask_gt]

            pred_np = pred.asnumpy() if len(pred) > 0 else np.zeros((0, 6))

            if len(pred) == 0:
                if len(gt_cls) > 0:
                    tp_empty = np.zeros((0, self.niou), dtype=bool)
                    self.metrics.update_stats(
                        tp_b=tp_empty, tp_m=tp_empty, conf=np.zeros(0), 
                        pred_cls=np.zeros(0), target_cls=gt_cls
                    )
                continue

            # 约束预测框溢出原图边际
            pred_np[:, :4] = np.clip(pred_np[:, :4], 0, imgsz)
            pred = ms.Tensor(pred_np, dtype=ms.float32)

            # 调用高度优化的底层算子，在浮点计算域还原实例级 Mask 蒙版
            masks_pred_bin = process_mask(proto[i], pred[:, 6:], pred[:, :4], (imgsz, imgsz), upsample=False)
            
            # 生成各个 IoU 阈值刻度对应的真阳性判断矩阵
            tp_b = self.calculate_tp(pred_np[:, :4], pred_np[:, 5], gt_bboxes, gt_cls, is_mask=False)
            tp_m = self.calculate_tp(masks_pred_bin, pred_np[:, 5], gt_masks, gt_cls, is_mask=True)

            self.metrics.update_stats(
                tp_b=tp_b, tp_m=tp_m, conf=pred_np[:, 4], 
                pred_cls=pred_np[:, 5], target_cls=gt_cls
            )

    def finalize_metrics(self):
        """执行最终度量计算"""
        if len(self.metrics.stats) == 0:
            LOGGER.warning("[WARNING] 当前评估轮次无有效度量样本输入。")
            self.stats = {"metrics/mAP50-95(B)": 0.0, "metrics/mAP50-95(M)": 0.0}
            return
            
        try:
            self.stats = self.metrics.process()
        except Exception as e:
            LOGGER.error(f"[ERROR] 指标聚合运算发生内部异常终止: {e}")
            self.stats = {"metrics/mAP50-95(B)": 0.0, "metrics/mAP50-95(M)": 0.0}

    def get_stats(self):
        """交还评估快照"""
        stats = super().get_stats()
        stats.update(self.stats)
        
        if self.metrics is not None:
            stats['fitness'] = self.metrics.fitness
        else:
            stats['fitness'] = 0.0
            
        self.results_dict = stats
        return stats

    def calculate_tp(self, preds, pred_cls, gts, gt_cls, is_mask=False):
        """
        全任务域通用匹配校验核心逻辑：解析 IoU 并构建布尔记录矩阵
        内置了预防因批尺寸特征不对齐引发的矩阵相乘 (MatMul) 异常保护
        """
        if len(gts) == 0:
            return np.zeros((preds.shape[0], self.niou), dtype=bool)

        p_tensor = ms.Tensor(preds) if not isinstance(preds, ms.Tensor) else preds
        g_tensor = ms.Tensor(gts) if not isinstance(gts, ms.Tensor) else gts
        pc_tensor = ms.Tensor(pred_cls) if not isinstance(pred_cls, ms.Tensor) else pred_cls
        gc_tensor = ms.Tensor(gt_cls) if not isinstance(gt_cls, ms.Tensor) else gt_cls

        if is_mask:
            n, h, w = p_tensor.shape
            m, gh, gw = g_tensor.shape
            
            # 若空间解析度与特征原型图失配，则强制将真实标签蒙版执行重采样收缩
            if (gh, gw) != (h, w):
                g_resized = ops.ResizeNearestNeighbor((h, w))(g_tensor.expand_dims(1))[:, 0]
                g = g_resized.view(m, -1).astype(ms.float32)
            else:
                g = g_tensor.view(m, -1).astype(ms.float32)
            
            p = p_tensor.view(n, -1).astype(ms.float32)
            
            inter = ops.matmul(p, g.T) 
            area_p = p.sum(1, keepdims=True)      
            area_g = g.sum(1, keepdims=True).T    
            iou = inter / (area_p + area_g - inter + 1e-7)
        else:
            g_xyxy = ms.Tensor(gts) if not isinstance(gts, ms.Tensor) else gts
            
            lt = ops.maximum(p_tensor[:, None, :2], g_xyxy[:, :2])
            rb = ops.minimum(p_tensor[:, None, 2:], g_xyxy[:, 2:])
            
            wh = ops.maximum(rb - lt, 0.0) 
            inter = wh[:, :, 0] * wh[:, :, 1]
            
            area_p = (p_tensor[:, 2] - p_tensor[:, 0]) * (p_tensor[:, 3] - p_tensor[:, 1])
            area_g = (g_xyxy[:, 2] - g_xyxy[:, 0]) * (g_xyxy[:, 3] - g_xyxy[:, 1])
            iou = inter / (area_p[:, None] + area_g - inter + 1e-7)

        correct_class = (pc_tensor.view(-1, 1) == gc_tensor.view(1, -1))
        
        tp = np.zeros((preds.shape[0], self.niou), dtype=bool)
        iou_np = iou.asnumpy()
        cc_np = correct_class.asnumpy()
        
        for i, threshold in enumerate(self.iou_v):
            match_matrix = (iou_np >= threshold) & cc_np
            tp[:, i] = match_matrix.any(axis=1)

        return tp