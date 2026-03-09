import sys
import os
import yaml
from pathlib import Path
import mindspore as ms
from mindspore import nn

# --- 导入底层架构和组件 ---
from engine.trainer import BaseTrainer
from models.yolo.classify.val import ClassificationValidator
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForClassification
from data.loaders import create_dataloader  
from utils.loss import YOLOClassificationLoss 
from utils.optimizer import build_optimizer, get_lr

class ClassificationTrainer(BaseTrainer):
    """图像分类任务专属 Trainer，实现基类的抽象方法"""

    def __init__(self, args):
        # 优先解析数据集配置 YAML，后续 DataLoader 与模型初始化需依赖此类信息
        with open(args.data, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
            
        # 启动基类的全局构建流程
        super().__init__(args)

    def get_dataloader(self, is_training):
        split_key = 'train' if is_training else 'val'
        dataset_path = os.path.join(self.data.get('path', ''), self.data[split_key])
        
        return create_dataloader(
            dataset_path, 
            imgsz=self.args.imgsz, 
            batch_size=self.args.batch, 
            is_training=is_training, 
            num_workers=self.args.workers
        )

    def get_model(self, cfg=None, weights=None):
        config = YOLOConfig(yaml_path=cfg, scale=self.args.scale, task='classify')
        config.nc = self.data.get('nc', 1000)
        model = YOLO11ForClassification(config)

        if weights and os.path.exists(weights):
            print(f"[INFO] 检测到预训练权重: {weights}，启动微调模式。")
            param_dict = ms.load_checkpoint(weights)
            
            # 分类任务微调时，通常需要剥离原有的分类输出层以适应新的类别数
            pop_keys = [k for k in param_dict.keys() if "model.10" in k]
            for key in pop_keys:
                param_dict.pop(key)
                
            ms.load_param_into_net(model, param_dict, strict_load=False)
            print("[INFO] 预训练 Backbone 参数加载完毕，已移除旧的分类头。")
            
        return model

    def build_optimizer(self, model):
        """生成学习率策略并构建优化器"""
        steps_per_epoch = self.train_loader.get_dataset_size()
        lr_list = get_lr(self.args, self.hyp, steps_per_epoch)
        return build_optimizer(model, lr_list, self.hyp)

    def get_loss_fn(self):
        return YOLOClassificationLoss()

    def get_validator(self):
        return ClassificationValidator(dataloader=self.val_loader, args=self.args)