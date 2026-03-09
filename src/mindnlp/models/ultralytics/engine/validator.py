import time
import logging
import yaml
from pathlib import Path

import mindspore as ms
from mindspore import ops
from tqdm import tqdm

from utils.ops import non_max_suppression

# 统一日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class BaseValidator:
    """
    验证任务基类
    负责管理验证数据集的加载、前向推理耗时统计，以及指标计算的评估等通用逻辑。
    """
    def __init__(self, dataloader=None, save_dir=None, args=None):
        self.args = args
        self.dataloader = dataloader
        self.save_dir = Path(save_dir) if save_dir else Path("./runs/val")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = ms.get_context("device_target")
        self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        self.seen = 0
        self.training = False

        # 解析超参数配置 (支持从 hyp.yaml 读取 conf, iou, half 等验证参数)
        self.hyp = {}
        if hasattr(args, 'hyp') and args.hyp and Path(args.hyp).exists():
            with open(args.hyp, "r", encoding="utf-8") as f:
                self.hyp = yaml.safe_load(f)

    def get_dataloader(self, dataset_path, batch_size):
        """抽象接口：获取验证数据集的 DataLoader"""
        raise NotImplementedError("子类必须实现 get_dataloader 方法")

    def __call__(self, model):
        """验证流程主入口"""
        self.training = model.training if hasattr(model, 'training') else False
        
        # 确保模型处于推理模式
        model.set_train(False) 
        self.init_metrics(model)

        bar = tqdm(self.dataloader.create_dict_iterator(), 
                   desc="Validating", 
                   total=self.dataloader.get_dataset_size())

        for batch_i, batch in enumerate(bar):
            # 1. 数据预处理
            t0 = time.time()
            batch = self.preprocess(batch)
            self.speed["preprocess"] += time.time() - t0

            # 2. 模型前向推理
            t1 = time.time()
            preds = model(batch["image"])
            self.speed["inference"] += time.time() - t1

            # 3. 后处理与指标收集
            t2 = time.time()
            preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            self.speed["postprocess"] += time.time() - t2

            self.seen += batch["image"].shape[0]

        # 4. 指标汇总与结果打印
        self.finalize_metrics()
        stats = self.get_stats()
        self.print_results()
        
        return stats

    def preprocess(self, batch):
        """
        通用图像预处理逻辑：类型转换与归一化
        """
        img = ops.cast(batch["image"], ms.float32)
        
        # 像素值归一化至 [0.0, 1.0] 域
        if img.max() > 2.0:
            img = img / 255.0
            
        batch["image"] = img
        return batch

    def postprocess(self, preds):
        """
        通用后处理逻辑（默认为目标检测任务的非极大值抑制 NMS）
        注意：分类等特定任务需在子类中重写此方法
        """
        p = preds[0] if isinstance(preds, tuple) else preds
        
        # 优先从超参数配置读取阈值，若无则回退至命令行参数或默认值
        conf_thres = self.hyp.get('conf', getattr(self.args, 'conf', 0.001))
        iou_thres = self.hyp.get('iou', getattr(self.args, 'iou', 0.6))
        
        out = non_max_suppression(
            p, 
            conf_thres=conf_thres, 
            iou_thres=iou_thres, 
            nc=getattr(self, 'nc', 80)
        )
        return out

    # --- 抽象接口定义 ---
    def init_metrics(self, model): pass
    def update_metrics(self, preds, batch): pass
    def finalize_metrics(self): pass
    
    def get_stats(self): 
        """计算平均单张图像的各阶段耗时 (ms)"""
        return {"speed": {k: v / self.seen * 1000 for k, v in self.speed.items() if self.seen > 0}}
    
    def print_results(self): 
        speed_str = " | ".join([f"{k}: {v:.1f}ms" for k, v in self.get_stats().get('speed', {}).items()])
        LOGGER.info(f"推理测速: {speed_str}")