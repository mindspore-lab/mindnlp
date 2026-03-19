import argparse
import os
import logging
import mindspore as ms

from models.yolo.segment.train import SegmentationTrainer

logging.basicConfig(level=logging.INFO, format='%(message)s')
LOGGER = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Instance Segmentation Training Pipeline")
    
    # 核心路径与配置
    parser.add_argument('--data', type=str, default='./cfg/datasets/coco128-seg.yaml', help='数据集结构定义 YAML')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-seg.yaml', help='模型拓扑配置文件')
    parser.add_argument('--hyp', type=str, default='./cfg/hyp.yaml', help='控制训练周期的超参数文件 (学习率、增强等)')
    parser.add_argument('--weights', type=str, default=None, help='预训练检查点路径 (.ckpt)')
    
    # 模型架构与执行流参数
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='基础模型复杂度层级')
    parser.add_argument('--epochs', type=int, default=100, help='训练总回合数')
    parser.add_argument('--batch_size', type=int, default=16, help='单路处理批次')
    parser.add_argument('--imgsz', type=int, default=640, help='进入网络的图像分辨率空间')
    parser.add_argument('--workers', type=int, default=8, help='多进程数据装载核心数')
    
    # 硬件与保存配置
    parser.add_argument('--device', type=str, default='Ascend', help='深度学习计算硬件靶标 (Ascend/GPU/CPU)')
    parser.add_argument('--val_interval', type=int, default=10, help='两次评估循环之间的 Epoch 间距')
    parser.add_argument('--save_dir', type=str, default='./runs/segment/train', help='训练档案与模型文件的输出')
    
    args = parser.parse_args()

    # 指配计算域
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 启动封装完善的对象生命周期
    LOGGER.info("[INFO] SegmentationTrainer 初始化就绪，即将进入训练大循环。")
    trainer = SegmentationTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()