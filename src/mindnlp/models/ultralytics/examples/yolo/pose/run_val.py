import argparse
import mindspore as ms
import os
import sys
import yaml

from models.yolo.pose.val import PoseValidator
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForPose
from data.loaders import create_dataloader

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Pose Estimation Standalone Validation")
    
    # --- 基础配置参数 ---
    parser.add_argument('--data', type=str, default='./cfg/datasets/coco8-pose.yaml', help='数据集配置文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-pose.yaml', help='模型架构 YAML 配置文件')
    parser.add_argument('--weights', type=str, default='./yolo11n-pose.ckpt', help='待评估的预训练权重文件路径')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型规模标识')
    
    # --- 评估环境参数 ---
    parser.add_argument('--imgsz', type=int, default=640, help='网络输入图像分辨率')
    parser.add_argument('--batch', type=int, default=4, help='验证过程的批次大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载并行线程数')
    parser.add_argument('--device', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], help='计算硬件目标')
    parser.add_argument('--save_dir', type=str, default='./runs/pose/val', help='验证结果与指标保存目录')
    
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 解析数据集配置
    with open(args.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        
    args.nc = data_cfg.get('nc', 1)
    kpt_shape = data_cfg.get('kpt_shape', [17, 3])
    names = data_cfg.get('names', {i: f'class_{i}' for i in range(args.nc)})
    val_dir = os.path.join(data_cfg.get('path', ''), data_cfg.get('val', 'val'))
    
    # 实例化数据加载器
    val_loader = create_dataloader(
        path=val_dir, 
        imgsz=args.imgsz, 
        batch_size=args.batch, 
        task='pose', 
        is_training=False, 
        num_workers=args.workers
    )

    # 实例化模型架构并加载权重
    config = YOLOConfig(yaml_path=args.model_cfg, scale=args.scale, task="pose")
    config.nc = args.nc
    config.kpt_shape = kpt_shape
    model = YOLO11ForPose(config)

    print(f"[Info] 加载验证权重: {args.weights}")
    ms.load_param_into_net(model, ms.load_checkpoint(args.weights), strict_load=False)

    # 执行验证流水线
    validator = PoseValidator(
        dataloader=val_loader, 
        save_dir=args.save_dir, 
        args=args, 
        names=names, 
        kpt_shape=kpt_shape
    )
    stats = validator(model=model)

    # 格式化输出核心评估指标
    print("\n" + "-"*50)
    print("[Result] 姿态估计验证完成。核心评估指标如下：")
    print(f" - [Box] 边界框 mAP@50-95:  {stats.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f" - [Pose] 关键点 mAP@50-95: {stats.get('metrics/mAP50-95(P)', 0.0):.4f}")
    print("-"*50 + "\n")

if __name__ == "__main__":
    main()