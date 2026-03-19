import logging
from types import SimpleNamespace

import mindspore as ms
from mindspore import ops

from engine.validator import BaseValidator
from data.loaders import create_dataloader
from utils.metrics import ClassifyMetrics

LOGGER = logging.getLogger(__name__)

class ClassificationValidator(BaseValidator):
    """图像分类任务验证器，继承并实现基类的相关接口"""

    def __init__(self, dataloader=None, save_dir=None, args=None):
        super().__init__(dataloader, save_dir, args)
        
        # 确保任务类型正确声明
        if self.args is None:
            self.args = SimpleNamespace(task="classify", half=False)
        else:
            self.args.task = "classify"
            
        self.names = None 

    def get_dataloader(self, dataset_path, batch_size=16):
        """构建分类任务专用 DataLoader"""
        return create_dataloader(
            path=dataset_path,
            imgsz=getattr(self.args, 'imgsz', 224),
            batch_size=batch_size,
            task='classify',
            is_training=False,
            num_workers=getattr(self.args, 'workers', 8)
        )

    def init_metrics(self, model):
        """初始化分类评价指标统计器"""
        self.names = getattr(model, 'names', {})
        self.nc = len(self.names) if self.names else 0

        self.metrics = ClassifyMetrics()
        self.targets = []
        self.preds = []

    def preprocess(self, batch):
        """分类任务的特殊预处理：支持半精度推理"""
        batch["image"] = ops.cast(batch["image"], ms.float32)
        
        # 若开启混合精度/半精度验证，转换数据类型
        use_half = self.hyp.get('half', getattr(self.args, 'half', False))
        if use_half:
            batch["image"] = ops.cast(batch["image"], ms.float16)
            
        return batch

    def postprocess(self, preds):
        """分类网络输出剥离，无需进行 NMS"""
        return preds[0] if isinstance(preds, (list, tuple)) else preds

    def update_metrics(self, preds, batch):
        """提取预测结果与真实标签，存储至列表以供后续指标计算"""
        target = batch.get("label", batch.get("cls"))
        
        # 获取置信度最高的 Top-5 索引
        n5 = min(preds.shape[1], 5)
        _, topk_indices = ops.topk(preds, n5, dim=1) 
        
        self.preds.append(topk_indices.asnumpy())
        self.targets.append(target.asnumpy())

    def get_stats(self):
        """触发 Metrics 类计算最终的验证指标，并合并统计结果"""
        self.metrics.process(self.targets, self.preds)
        metrics_dict = self.metrics.results_dict 
        
        stats = super().get_stats() if hasattr(super(), 'get_stats') else {}
        stats.update(metrics_dict)
        return stats

    def print_results(self):
        super().print_results()
        stats = self.get_stats()
        LOGGER.info(
            f"验证结果 | Top-1 Acc: {stats.get('metrics/accuracy_top1', 0):.4f} | "
            f"Top-5 Acc: {stats.get('metrics/accuracy_top5', 0):.4f}"
        )