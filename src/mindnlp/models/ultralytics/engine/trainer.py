import os
import yaml
from pathlib import Path
import mindspore as ms
import numpy as np
from mindspore import nn, ops
from utils.ema import ModelEMA

class TrainStepWithClip(nn.TrainOneStepCell):
    """支持梯度裁剪的单步训练封装类"""
    def __init__(self, network, optimizer, clip_val=10.0):
        super(TrainStepWithClip, self).__init__(network, optimizer)
        self.clip_val = clip_val
        self.grad_fn = ops.value_and_grad(self.network, grad_position=None, weights=self.weights)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        # 限制全局梯度范数，防止梯度爆炸
        grads = ops.clip_by_global_norm(grads, self.clip_val)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

class YOLOWithLossCell(nn.Cell):
    """适配检测任务的多输入 Loss 封装，负责连接模型输出与损失函数"""
    def __init__(self, backbone, loss_fn):
        super(YOLOWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, labels):
        #  前向传播获取预测结果
        pred = self._backbone(x)
        #  将预测值与完整的标签字典传给损失函数，防止维度丢失
        return self._loss_fn(pred, labels)

class BaseTrainer:
    """训练基类：负责调度全局流程、解析配置、管理模型与日志保存"""
    
    def __init__(self, args):
        self.args = args
        self.device = ms.get_context("device_target")
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_fitness = 0.0
        self.start_epoch = 0
        self.epochs = args.epochs

        # 统一加载超参数配置 (hyp.yaml)
        if hasattr(args, 'hyp') and os.path.exists(args.hyp):
            with open(args.hyp, "r", encoding="utf-8") as f:
                self.hyp = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"[ERROR] 未找到超参数配置文件: {getattr(args, 'hyp', '未指定')}")

        # 调度子类构建核心组件
        self.train_loader = self.get_dataloader(is_training=True)
        self.val_loader = self.get_dataloader(is_training=False)
        self.model = self.get_model(cfg=self.args.model_cfg, weights=self.args.weights)
        self.optimizer = self.build_optimizer(self.model)
        self.ema = ModelEMA(self.model)
        
        # 构建训练计算图
        self.loss_fn = self.get_loss_fn()
        self.net_with_loss = YOLOWithLossCell(self.model, self.loss_fn)
    
        self.train_step = TrainStepWithClip(self.net_with_loss, self.optimizer, clip_val=10.0)

    def train(self):
        """核心训练迭代逻辑"""
        print(f"[INFO] 训练任务启动，总轮数: {self.epochs} epochs")
        steps_per_epoch = self.train_loader.get_dataset_size()

        for epoch in range(self.start_epoch, self.epochs):
            self.model.set_train(True)
            self.train_step.set_train(True)
            
            for step, batch in enumerate(self.train_loader.create_dict_iterator()):
                # 预处理批次数据
                batch = self.preprocess_batch(batch)
                
                # 执行单步训练，batch 字典包含所需的所有标签信息
                imgs = batch["image"]
                loss = self.train_step(imgs, batch) 
                self.ema.update(self.model)
                
                # 日志打印逻辑
                if step % 10 == 0:
                    if isinstance(loss, (tuple, list)):
                        loss_items = [float(x.asnumpy()) for x in loss]
                        num_loss = len(loss_items)
                        
                        if num_loss == 4:  # 分割任务
                            l_box, l_cls, l_dfl, l_mask = loss_items
                            total_loss = sum(loss_items)
                            loss_str = f"Box: {l_box:.4f} | Cls: {l_cls:.4f} | DFL: {l_dfl:.4f} | Mask: {l_mask:.4f}"
                        elif num_loss == 3:  # 检测任务
                            l_box, l_cls, l_dfl = loss_items
                            total_loss = sum(loss_items)
                            loss_str = f"Box: {l_box:.4f} | Cls: {l_cls:.4f} | DFL: {l_dfl:.4f}"
                        else: 
                            total_loss = sum(loss_items)
                            loss_str = f"Loss Items: {[f'{x:.4f}' for x in loss_items]}"
                            
                        print(f"Epoch [{epoch}/{self.epochs-1}] Step [{step}/{steps_per_epoch}] | Total Loss: {total_loss:.4f} | {loss_str}")
                    else:
                        loss_val = float(loss.asnumpy()) if hasattr(loss, "asnumpy") else float(loss)
                        print(f"Epoch [{epoch}/{self.epochs-1}] Step [{step}/{steps_per_epoch}] | Loss: {loss_val:.4f}")

            # 验证与保存阶段
            if (epoch + 1) % self.args.val_interval == 0 or epoch == self.epochs - 1:
                print(f"\n[INFO] 开始执行 Epoch {epoch} 验证程序...")
                self.model.set_train(False) 
                
                validator = self.get_validator()
                stats = validator(self.ema.ema_model)
                
                print("-" * 50)
                print(f"[评估报告] Epoch {epoch}")
                for k, v in stats.items():
                    if k.startswith('metrics/'):
                        metric_name = k.replace('metrics/', '')
                        print(f"  - {metric_name:<15} : {float(v):.5f}")
                
                fitness_f = float(stats.get('fitness', 0.0))
                print(f"[INFO] 当前模型综合评价指标 (Fitness): {fitness_f:.5f}")
                print("-" * 50 + "\n")
                
                self._save_checkpoint(epoch, fitness_f)
                
                self.model.set_train(True) 
                self.train_step.set_train(True)

    def _save_checkpoint(self, epoch, fitness):
        """权重序列化保存逻辑"""
        ms.save_checkpoint(self.ema.ema_model, str(self.save_dir / "last.ckpt"))
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            ms.save_checkpoint(self.ema.ema_model, str(self.save_dir / "best.ckpt"))
            print(f"[INFO] 已更新最佳模型权重 (best.ckpt)，当前最高精度: {self.best_fitness:.4f}")

    # 抽象接口声明，子类需实现具体的业务逻辑
    def get_dataloader(self, is_training): raise NotImplementedError
    def get_model(self, cfg, weights): raise NotImplementedError
    def build_optimizer(self, model): raise NotImplementedError
    def get_validator(self): raise NotImplementedError
    def get_loss_fn(self): raise NotImplementedError
    
    def preprocess_batch(self, batch):
        """
        数据预处理：执行图像归一化与标签 Tensor 转换
        """
        # 图像预处理：强制转为 Tensor 并执行像素归一化
        img = batch["image"]
        if not isinstance(img, ms.Tensor):
            img = ms.Tensor(img, ms.float32)
        
        # 仅在数据为原始像素值 (0-255) 时进行归一化
        if img.max() > 1.0:
            img = ops.cast(img, ms.float32) / 255.0
        batch["image"] = img

        # 标签处理：确保 bboxes 和 batch_idx 类型符合 Loss 计算要求
        if "bboxes" in batch and "batch_idx" in batch:
            if not isinstance(batch["bboxes"], ms.Tensor):
                batch["bboxes"] = ms.Tensor(batch["bboxes"], ms.float32)
            
            if not isinstance(batch["batch_idx"], ms.Tensor):
                batch["batch_idx"] = ms.Tensor(batch["batch_idx"], ms.int32)
                
        return batch