import argparse
import os
import yaml
import mindspore as ms

from models.yolo.detect.train import DetectionTrainer

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Object Detection Training Pipeline")
    
    # ---------------- 核心执行参数 ----------------
    parser.add_argument('--data', type=str, default='./cfg/datasets/coco128.yaml', help='数据集配置文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11.yaml', help='模型拓扑架构配置文件')
    parser.add_argument('--hyp', type=str, default='./cfg/hyp.yaml', help='超参数配置文件 (包含学习率、增强策略等)')
    parser.add_argument('--weights', type=str, default='', help='预训练 Checkpoint 路径 (留空则随机初始化)')
    
    # ---------------- 训练规模参数 ----------------
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型规模变体')
    parser.add_argument('--epochs', type=int, default=100, help='训练总轮数')
    parser.add_argument('--batch', type=int, default=16, help='单卡批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='网络输入分辨率')
    parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    
    # ---------------- 环境与保存机制 ----------------
    parser.add_argument('--val_interval', type=int, default=10, help='验证评估频率 (单位: Epoch)')
    parser.add_argument('--save_dir', type=str, default='./runs/detect/train', help='模型与日志保存目录')
    parser.add_argument('--device', type=str, default='Ascend', help='计算硬件 (Ascend/GPU/CPU)')
    
    args = parser.parse_args()

    # 配置 MindSpore 运行模式与硬件靶标
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 动态解析数据集类别总数
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"[ERROR] 无法找到数据集配置文件: {args.data}")
        
    with open(args.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    args.nc = data_cfg.get('nc', 80) 
    
    # 实例化并触发训练调度器
    trainer = DetectionTrainer(args)
    trainer.train()

if __name__ == "__main__":
    main()