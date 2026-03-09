import argparse
import mindspore as ms
import os
import yaml

from models.yolo.detect.val import DetectionValidator
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForObjectDetection
from data.loaders import create_dataloader

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Object Detection Standalone Validation")
    
    # --- 核心验证参数 ---
    parser.add_argument('--data', type=str, default='./cfg/datasets/coco128.yaml', help='数据集配置文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11.yaml', help='模型架构 YAML 配置文件')
    parser.add_argument('--weights', type=str, default='./yolo11n.ckpt', help='待验证的预训练权重文件路径')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型规模标识')
    
    # --- 运行环境参数 ---
    parser.add_argument('--imgsz', type=int, default=640, help='验证测试所用的图像分辨率')
    parser.add_argument('--batch', type=int, default=16, help='数据加载过程的批次大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载的并行子线程数')
    parser.add_argument('--device', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], help='目标计算硬件平台')
    parser.add_argument('--save_dir', type=str, default='./runs/detect/val', help='验证日志及性能指标的保存目录')
    
    args = parser.parse_args()

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 解析数据集配置
    print(f"[Info] 载入数据集配置信息: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        
    args.nc = data_cfg.get('nc', 80)
    data_path = data_cfg.get('path', '')
    val_dir = os.path.join(data_path, data_cfg.get('val', 'val'))
    
    print("[Info] 构建目标检测验证数据集 DataLoader...")
    val_loader = create_dataloader(
        path=val_dir, 
        imgsz=args.imgsz, 
        batch_size=args.batch, 
        task='detect', 
        is_training=False, 
        num_workers=args.workers
    )

    #  实例化模型架构并加载权重
    print(f"[Info] 部署模型计算图架构 (Scale: {args.scale}, Classes: {args.nc})")
    config = YOLOConfig(yaml_path=args.model_cfg, scale=args.scale, task="detect")
    config.nc = args.nc
    model = YOLO11ForObjectDetection(config)

    if os.path.exists(args.weights):
        print(f"[Info] 加载模型参数权重: {args.weights}")
        param_dict = ms.load_checkpoint(args.weights)
        ms.load_param_into_net(model, param_dict, strict_load=False)
    else:
        raise FileNotFoundError(f"[Error] 指定的权重文件不存在: {args.weights}，请验证输入路径。")

    # 实例化 Validator 执行验证评估
    print("[Info] 触发验证评估流水线执行过程...")
    validator = DetectionValidator(dataloader=val_loader, save_dir=args.save_dir, args=args)
    stats = validator(model=model)

    mAP50 = stats.get('metrics/mAP50(B)', 0.0)
    mAP50_95 = stats.get('metrics/mAP50-95(B)', 0.0)
    
    print("\n" + "-" * 50)
    print("[Result] 目标检测验证任务结束。全局性能指标如下：")
    print(f"   - mAP@50:      {mAP50:.4f}")
    print(f"   - mAP@50-95:   {mAP50_95:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()