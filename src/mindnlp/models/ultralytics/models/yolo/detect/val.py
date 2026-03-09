import logging
import numpy as np
import mindspore as ms
from mindspore import ops

from engine.validator import BaseValidator
from utils.metrics import DetMetrics, process_batch
from utils.ops import non_max_suppression, xywh2xyxy_np
from data.loaders import create_dataloader

LOGGER = logging.getLogger(__name__)

class DetectionValidator(BaseValidator):
    """
    目标检测任务专用验证器
    继承 BaseValidator
    """
    def __init__(self, dataloader=None, save_dir=None, args=None):
        super().__init__(dataloader, save_dir, args)
        # 初始化 COCO 标准的 10 个 IoU 评估阈值 (0.50 到 0.95，步长 0.05)
        self.iou_v = np.linspace(0.5, 0.95, 10) 
        self.metrics = None
        self.names = None
        self.stats = {} 

    def get_dataloader(self, dataset_path, batch_size=16):
        """
        构建目标检测任务数据流
        使用 is_training=False 关闭随机数据增强
        """
        return create_dataloader(
            path=dataset_path,
            imgsz=getattr(self.args, 'imgsz', 640),
            batch_size=batch_size,
            task='detect',  
            is_training=False, 
            num_workers=getattr(self.args, 'workers', 8)
        )

    def init_metrics(self, model):
        """初始化评估指标计算器与类别映射表"""
        self.names = getattr(model, 'names', {i: f'class_{i}' for i in range(getattr(self.args, 'nc', 80))})
        self.metrics = DetMetrics(names=self.names)

    def preprocess(self, batch):
        """图像预处理：标准化像素值域至 [0, 1]"""
        img_key = "img" if "img" in batch else "image"
        img = ops.cast(batch[img_key], ms.float32) / 255.0
        batch[img_key] = img
        return batch

    def postprocess(self, preds):
        """
        验证期后处理：执行非极大值抑制 (NMS)
        注：为计算完整的 Precision-Recall 曲线，验证阶段的 conf_thres 必须设定为极小值 (如 0.001)
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        # IoU 阈值由 hyp.yaml 控制，置信度强制为 0.001 以计算 mAP
        iou_thres = self.hyp.get('iou', 0.6)
        preds_nms = non_max_suppression(preds, conf_thres=0.001, iou_thres=iou_thres)
        return preds_nms

    def update_metrics(self, preds, batch):
        """对比预测边界框与真实标签，统计各个阈值下的 True Positives"""
        batch_idx = batch["batch_idx"].reshape(-1).asnumpy()
        targets_bboxes = batch["bboxes"].asnumpy() 
        targets_cls = batch["cls"].reshape(-1).asnumpy()

        for si, pred in enumerate(preds):
            # 1. 提取当前图像的 Ground Truth
            mask = (batch_idx == si)
            t_cls = targets_cls[mask]
            t_box = targets_bboxes[mask]
            
            # 2. 构建格式化标签矩阵 [N, 5] -> [class, x1, y1, x2, y2]
            if len(t_cls) > 0:
                imgsz = getattr(self.args, 'imgsz', 640)
                t_box_xyxy = xywh2xyxy_np(t_box) * imgsz
                labels = np.concatenate((t_cls[:, None], t_box_xyxy), axis=1)
            else:
                labels = np.zeros((0, 5))

            # 3. 提取当前图像的预测结果
            p_det = pred.asnumpy() if len(pred) > 0 else np.zeros((0, 6))


            # 4. 极端情况处理：无预测框
            if len(p_det) == 0:
                if len(labels):
                    self.metrics.update_stats(np.zeros((0, 10), dtype=bool), np.zeros(0), np.zeros(0), labels[:, 0])
                continue

            # 5. 执行 IoU 匹配，计算 TP 矩阵
            tp = process_batch(p_det, labels, self.iou_v)
            self.metrics.update_stats(tp, p_det[:, 4], p_det[:, 5], labels[:, 0])

    def finalize_metrics(self):
        """计算最终聚合的 mAP 指标"""
        self.stats = self.metrics.process()

    def get_stats(self):
        """整合信息与验证指标并返回"""
        stats = super().get_stats()
        stats.update(self.stats)
        
        # 兼容 Trainer 生命周期中的模型保存判定逻辑
        stats['fitness'] = stats.get('metrics/mAP50-95(B)', 0.0)
        self.results_dict = stats
        
        return stats