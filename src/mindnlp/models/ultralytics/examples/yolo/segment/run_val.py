import argparse
import mindspore as ms
import os
import yaml

from models.yolo.segment.val import SegmentationValidator
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForSegmentation
from data.loaders import create_dataloader

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Segmentation Standalone Validation")
    
    # --- 核心验证参数 ---
    parser.add_argument('--data', type=str, default='./cfg/datasets/coco128-seg.yaml', help='数据集配置文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-seg.yaml', help='模型架构 YAML 配置文件')
    parser.add_argument('--weights', type=str, default='./yolo11n-seg.ckpt', help='待评估的预训练权重文件路径')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型规模标识')
    
    # --- 运行环境参数 ---
    parser.add_argument('--imgsz', type=int, default=640, help='网络推理输入分辨率')
    parser.add_argument('--batch', type=int, default=16, help='验证过程的批次大小')
    parser.add_argument('--workers', type=int, default=8, help='数据加载并行线程数')
    parser.add_argument('--device', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], help='计算硬件目标')
    parser.add_argument('--save_dir', type=str, default='./runs/segment/val', help='验证结果与指标保存目录')
    
    args = parser.parse_args()

    # 设置 MindSpore 运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 解析数据集配置，准备 DataLoader
    print(f"[Info] 解析数据集配置: {args.data}")
    with open(args.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        
    args.nc = data_cfg.get('nc', 80)
    names = data_cfg.get('names', {i: f'class_{i}' for i in range(args.nc)})
    
    data_path = data_cfg.get('path', '')
    val_dir = os.path.join(data_path, data_cfg.get('val', 'val'))
    
    print("[Info] 正在构建验证集 DataLoader...")
    val_loader = create_dataloader(
        path=val_dir, 
        imgsz=args.imgsz, 
        batch_size=args.batch, 
        task='segment', 
        is_training=False, 
        num_workers=args.workers
    )

    # 实例化模型并加载权重
    print(f"[Info] 初始化分割模型架构 (Scale: {args.scale}, Classes: {args.nc})")
    config = YOLOConfig(yaml_path=args.model_cfg, scale=args.scale, task="segment")
    config.nc = args.nc
    model = YOLO11ForSegmentation(config)

    if os.path.exists(args.weights):
        print(f"[Info] 正在加载验证权重: {args.weights}")
        param_dict = ms.load_checkpoint(args.weights)
        ms.load_param_into_net(model, param_dict, strict_load=False)
    else:
        raise FileNotFoundError(f"[Error] 找不到权重文件: {args.weights}，请检查路径。")

    # 实例化 Validator 并执行验证
    print("[Info] 启动分割验证流水线...")
    validator = SegmentationValidator(dataloader=val_loader, save_dir=args.save_dir, args=args, names=names)
    stats = validator(model=model)

    # 提取双重指标 (Bounding Box & Mask)
    box_mAP50 = stats.get('metrics/mAP50(B)', 0.0)
    box_mAP50_95 = stats.get('metrics/mAP50-95(B)', 0.0)
    mask_mAP50 = stats.get('metrics/mAP50(M)', 0.0)
    mask_mAP50_95 = stats.get('metrics/mAP50-95(M)', 0.0)
    
    print("\n" + "-"*50)
    print("[Result] 分割验证任务完成。核心指标如下：")
    print("  [Box 目标检测]")
    print(f"   - mAP@50:      {box_mAP50:.4f}")
    print(f"   - mAP@50-95:   {box_mAP50_95:.4f}")
    print("  [Mask 实例分割]")
    print(f"   - mAP@50:      {mask_mAP50:.4f}")
    print(f"   - mAP@50-95:   {mask_mAP50_95:.4f}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()