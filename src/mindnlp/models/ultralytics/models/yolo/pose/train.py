import os
import yaml
import mindspore as ms
from mindspore import nn, ops

from engine.trainer import BaseTrainer
from modeling_yolo import YOLO11ForPose, YOLOConfig
from utils.loss import v8PoseLoss
from utils.optimizer import build_optimizer, get_lr
from data.loaders import create_dataloader

class PoseTrainer(BaseTrainer):
    """
    姿态估计(Pose)任务专属 Trainer
    继承自 BaseTrainer，封装了数据加载、模型初始化、损失构建及优化器配置等底层训练循环
    """
    def __init__(self, args):
        # 解析数据集 YAML 获取类别数和关键点形状
        with open(args.data, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
            
        # 解析超参数 YAML (hyp.yaml)
        self.hyp = {}
        if hasattr(args, 'hyp') and args.hyp and os.path.exists(args.hyp):
            with open(args.hyp, 'r', encoding='utf-8') as f:
                self.hyp = yaml.safe_load(f)
                
        super().__init__(args)
        
        # 姿态估计专属的损失函数名称集 (包含边界框回归、分类、分布焦点、关键点坐标及置信度)
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss", "pose_loss", "kobj_loss"]

    def get_dataloader(self, is_training):
        """实例化并返回姿态估计专属的 DataLoader"""
        split_key = 'train' if is_training else 'val'
        dataset_path = os.path.join(self.data.get('path', ''), self.data[split_key])
        
        return create_dataloader(
            dataset_path, 
            imgsz=self.args.imgsz, 
            batch_size=getattr(self.args, 'batch_size', self.args.batch), 
            task='pose',
            is_training=is_training, 
            num_workers=getattr(self.args, 'workers', 8)
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """模型结构构建与预训练权重加载验证"""
        config = YOLOConfig(yaml_path=self.args.model_cfg, scale=self.args.scale, task='pose')
        
        # 动态配置 Pose 专属属性
        config.nc = self.data.get('nc', 1)  # 姿态估计通常默认为 1 个类 (person)
        config.kpt_shape = self.data.get('kpt_shape', [17, 3])
        model = YOLO11ForPose(config)

        # 补充 Loss 计算依赖的核心属性，防止前向传播期间出现 AttributeError
        model.nc = config.nc
        model.kpt_shape = config.kpt_shape
        model.reg_max = getattr(config, 'reg_max', 16)
        model.stride = ms.Tensor([8, 16, 32], dtype=ms.float32)

        if weights and os.path.exists(weights):
            if verbose:
                print(f"[Info] 加载预训练权重进行微调: {weights}")
            param_dict = ms.load_checkpoint(weights)
            
            # 开启 strict_load=False 以允许部分权重不匹配 (适用于网络结构微调)
            param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict, strict_load=False)

            if verbose:
                print("-" * 50)
                print("[Info] 预训练权重加载完整性校验：")
                print(f" - 模型总参数量: {len(model.parameters_dict())}")
                print(f" - 未能加载的参数数量: {len(param_not_load)}")
                if len(param_not_load) > 0:
                    print(f"[Warning] 存在 {len(param_not_load)} 个参数未成功匹配。")
                    print(f" - 未加载参数示例: {param_not_load[:5]}") 
                print("-" * 50)
            
        return model

    def build_optimizer(self, model):
        """基于 hyp.yaml 或默认配置构建学习率策略与优化器"""
        steps_per_epoch = self.train_loader.get_dataset_size()
        lr_list = get_lr(self.args, self.hyp, steps_per_epoch)
        return build_optimizer(model, lr_list, self.hyp)

    def get_loss_fn(self):
        """实例化姿态估计专用损失函数"""
        criterion = v8PoseLoss(self.model)
        criterion.imgsz = self.args.imgsz
        return criterion

    def get_validator(self):
        """实例化姿态验证器，用于训练周期间的性能评估"""
        from models.yolo.pose.val import PoseValidator
        
        self.args.task = 'pose'
        self.args.mode = 'val'
        
        names = self.data.get('names', {i: f'class_{i}' for i in range(self.data.get('nc', 1))})
        kpt_shape = self.data.get('kpt_shape', [17, 3])
        
        return PoseValidator(
            dataloader=self.val_loader, 
            save_dir=self.save_dir, 
            args=self.args, 
            names=names,
            kpt_shape=kpt_shape
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """格式化输出损失字典，便于日志系统记录"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        return keys