import os
import yaml
import logging
import mindspore as ms

from modeling_yolo import YOLO11ForObjectDetection
from configuration_yolo import YOLOConfig
from data.loaders import create_dataloader 
from engine.trainer import BaseTrainer
from utils.optimizer import build_optimizer, get_lr
from utils.loss import v8DetectionLoss 

# 统一日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class DetectionTrainer(BaseTrainer):
    """
    目标检测任务专用 Trainer，继承并实现基类抽象方法
    """
    def __init__(self, args):
        super().__init__(args=args)
        # 预定义检测任务的三大损失项，用于日志格式化
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]

    def get_dataloader(self, is_training=True):
        """
        构建目标检测专用的数据流水线
        """    
        with open(self.args.data, 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
            
        split = 'train' if is_training else 'val'
        base_path = data_cfg.get('path', '')
        sub_path = data_cfg.get(split, '')
        dataset_path = os.path.join(base_path, sub_path) if base_path else sub_path

        return create_dataloader(
            path=dataset_path,
            imgsz=getattr(self.args, 'imgsz', 640),
            batch_size=getattr(self.args, 'batch', 16),
            task='detect',  
            is_training=is_training,
            num_workers=getattr(self.args, 'workers', 8),
            hyp=self.hyp  
        )

    def build_optimizer(self, model):
        """
        基于统一调度策略构建优化器与学习率衰减列表
        """
        steps_per_epoch = self.train_loader.get_dataset_size()
        lr_list = get_lr(self.args, self.hyp, steps_per_epoch)
        return build_optimizer(model, lr_list, self.hyp)

    def get_model(self, cfg=None, weights=None):
        """
        构建目标检测网络拓扑并对齐预训练权重
        """
        config = YOLOConfig(yaml_path=self.args.model_cfg, scale=self.args.scale, task='detect')
        
        if hasattr(self.args, 'nc'):
            config.nc = self.args.nc
            
        model = YOLO11ForObjectDetection(config)
        
        # 为网络注入 Loss 计算所需的核心架构常数
        model.nc = config.nc
        model.reg_max = getattr(config, 'reg_max', 16)
        model.stride = ms.Tensor([8, 16, 32], dtype=ms.float32)

        if weights and os.path.exists(weights):
            LOGGER.info(f"正在加载预训练权重: {weights}")
            param_dict = ms.load_checkpoint(weights)
            param_not_load, _ = ms.load_param_into_net(model, param_dict, strict_load=False)
            
            # 生成标准的权重审计日志
            LOGGER.info("-" * 40)
            LOGGER.info("[权重加载审计报告]")
            LOGGER.info(f"模型参数总量: {len(model.trainable_params())}")
            LOGGER.info(f"未匹配/未加载参数数量: {len(param_not_load)}")
            if len(param_not_load) > 0:
                LOGGER.info("未匹配参数抽样清单:")
                for p in param_not_load[:10]:  
                    LOGGER.info(f"  -> {p}")
            LOGGER.info("-" * 40)
            
        return model

    def get_loss_fn(self):
        """
        实例化目标检测专用损失计算图
        提取检测头 (Detect Head) 作为基础组件传递
        """
        model_head = self.model.model[-1]
        return v8DetectionLoss(model_head)

    def get_validator(self):
        """
        实例化目标检测验证器
        """
        from models.yolo.detect.val import DetectionValidator
        
        # 净化传递给验证器的配置字典，剔除不相关或由 yaml 管理的键值
        args_dict = vars(self.args).copy()
        custom_keys = ['nc', 'model_cfg', 'scale', 'weights']
        for key in custom_keys:
            args_dict.pop(key, None)
            
        args_dict['mode'] = 'val'
        args_dict['task'] = 'detect'
        
        return DetectionValidator(dataloader=self.val_loader, save_dir=self.save_dir, args=self.args)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """格式化输出 Loss 指标"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys