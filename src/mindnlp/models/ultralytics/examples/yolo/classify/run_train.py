import argparse
import mindspore as ms
import sys
import os

# 从核心库中导入纯净的 Trainer
from models.yolo.classify.train import ClassificationTrainer

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Classification Training")
    
    parser.add_argument('--data', type=str, default='./cfg/datasets/imagenette2-160.yaml')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-cls.yaml')
    parser.add_argument('--weights', type=str, default='./yolo11n-cls.ckpt', help="初始化权重路径，留空则从头训练")
    parser.add_argument('--scale', type=str, default='n')
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8, help="数据加载的线程数")
    parser.add_argument('--device', type=str, default='Ascend')
    parser.add_argument('--val_interval', type=int, default=1, help="每隔几个 epoch 验证一次")
    parser.add_argument('--save_dir', type=str, default="./runs/cls/train")

    
    parser.add_argument('--hyp', type=str, default='./cfg/hyp.yaml', help="算法调优超参数配置文件路径")
    
    args = parser.parse_args()
    
    # 设置运行模式与硬件
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 面向对象启动
    trainer = ClassificationTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()