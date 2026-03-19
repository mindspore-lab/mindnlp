import argparse
import os
import mindspore as ms

from models.yolo.pose.train import PoseTrainer

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Pose Estimation Training Pipeline")
    
    # 1. 基础与架构参数
    parser.add_argument('--data', type=str, default='./cfg/datasets/coco8-pose.yaml', help='数据集配置文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-pose.yaml', help='模型架构 YAML 配置文件')
    parser.add_argument('--hyp', type=str, default='./cfg/hyp.yaml', help='训练超参数配置文件 (学习率、衰减等)')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='网络规模标识')
    parser.add_argument('--weights', type=str, default='./yolo11n-pose.ckpt', help='预训练权重文件路径 (.ckpt)')
    
    # 2. 训练周期与计算超参数
    parser.add_argument('--epochs', type=int, default=100, help='总训练迭代轮数')
    parser.add_argument('--batch', type=int, default=4, help='全局批处理大小')
    parser.add_argument('--imgsz', type=int, default=640, help='网络输入图像分辨率')
    parser.add_argument('--workers', type=int, default=8, help='数据加载并行线程数')
    parser.add_argument('--device', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], help='目标计算硬件平台')
    parser.add_argument('--val_interval', type=int, default=10, help='验证评估频率 (单位: epoch)')
    parser.add_argument('--save_dir', type=str, default='./runs/pose/train', help='权重及日志保存输出目录')
    
    args = parser.parse_args()

    # 配置 MindSpore 运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 面向对象启动训练流水线
    trainer = PoseTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()