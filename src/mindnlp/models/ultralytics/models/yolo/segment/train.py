import os
import yaml
import logging
import mindspore as ms

# --- 核心架构组件 ---
from engine.trainer import BaseTrainer
from modeling_yolo import YOLO11ForSegmentation, YOLOConfig
from utils.loss import v8SegmentationLoss
from utils.optimizer import build_optimizer, get_lr
from data.loaders import create_dataloader

LOGGER = logging.getLogger(__name__)

class SegmentationTrainer(BaseTrainer):
    """
    实例分割任务专属 Trainer
    继承 BaseTrainer 以复用标准的训练循环
    """
    def __init__(self, args):
        with open(args.data, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
            
        super().__init__(args)
        # 定义分割任务独有的 4 项损失函数
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss", "seg_loss"]

    def get_dataloader(self, is_training):
        """挂载分割任务特有的 DataLoader"""
        split_key = 'train' if is_training else 'val'
        dataset_path = os.path.join(self.data.get('path', ''), self.data[split_key])
        
        return create_dataloader(
            dataset_path, 
            imgsz=self.args.imgsz, 
            batch_size=self.args.batch_size, 
            task='segment',
            is_training=is_training, 
            num_workers=getattr(self.args, 'workers', 8),
            hyp=self.hyp  # 传递超参数字典以驱动数据增强
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """构建分割模型拓扑并加载检查点"""
        config = YOLOConfig(yaml_path=self.args.model_cfg, scale=self.args.scale, task='segment')
        config.nc = self.data.get('nc', 80)
        model = YOLO11ForSegmentation(config)

        # 注入辅助特征解码的核心张量属性
        model.nc = config.nc
        model.reg_max = getattr(config, 'reg_max', 16)
        model.stride = ms.Tensor([8, 16, 32], dtype=ms.float32)

        if weights and os.path.exists(weights):
            LOGGER.info(f"正在载入初始检查点权重: {weights}")
            param_dict = ms.load_checkpoint(weights)
            
            # 使用 strict_load=False 容忍迁移学习阶段的分类头维度差异
            param_not_load, _ = ms.load_param_into_net(model, param_dict, strict_load=False)

            LOGGER.info("-" * 40)
            LOGGER.info("[权重匹配核查报告]")
            LOGGER.info(f"待加载参数总量: {len(param_dict)}")
            LOGGER.info(f"网络未被初始化的参数数目: {len(param_not_load)}")
            
            # 若缺失参数过多 (通常由于配置失误或网络拓扑更改导致)，可依据具体环境抛出警告
            if len(param_not_load) > 0:
                LOGGER.warning("部分网络参数未能在 Checkpoint 中寻址匹配，样本如下:")
                for p in param_not_load[:5]:
                    LOGGER.warning(f"  -> {p}")
            LOGGER.info("-" * 40)
            
        return model

    def build_optimizer(self, model):
        """基于 hyp.yaml 或默认配置构建学习率策略与优化器"""
        steps_per_epoch = self.train_loader.get_dataset_size()
        lr_list = get_lr(self.args, self.hyp, steps_per_epoch)
        return build_optimizer(model, lr_list, self.hyp)

    def get_loss_fn(self):
        """配置支持 Prototype 掩码评估的损失计算图"""
        criterion = v8SegmentationLoss(self.model)
        criterion.imgsz = self.args.imgsz
        return criterion

    def get_validator(self):
        """装载针对 Mask mAP 计算定制的实例分割验证器"""
        from models.yolo.segment.val import SegmentationValidator
        
        args_dict = vars(self.args)
        args_dict['task'] = 'segment'
        args_dict['mode'] = 'val'
        
        names = self.data.get('names', {i: f'class_{i}' for i in range(self.data.get('nc', 80))})
        return SegmentationValidator(dataloader=self.val_loader, save_dir=self.save_dir, args=self.args, names=names)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys