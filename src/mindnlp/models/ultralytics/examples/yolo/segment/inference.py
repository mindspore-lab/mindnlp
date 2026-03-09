import argparse
import mindspore as ms
import sys
import os

from models.yolo.segment.predict import SegmentationPredictor
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForSegmentation

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Segmentation Inference Pipeline")
    
    # --- 核心推理输入与架构参数 ---
    parser.add_argument('--source', type=str, default='./datasets/coco128-seg/images/train2017', help='待推理图像的路径或文件夹目录')
    parser.add_argument('--weights', type=str, default='./yolo11n-seg.ckpt', help='预训练权重文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-seg.yaml', help='模型架构 YAML 配置文件')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型规模标识')
    parser.add_argument('--nc', type=int, default=80, help='预测的类别总数')
    
    # --- 推理后处理与阈值参数 ---
    parser.add_argument('--imgsz', type=int, default=640, help='网络推理输入分辨率')
    parser.add_argument('--conf', type=float, default=0.25, help='预测框置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS 交并比 (IoU) 阈值')
    
    # --- 运行环境配置 ---
    parser.add_argument('--device', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], help='计算硬件目标')
    parser.add_argument('--save_dir', type=str, default='./runs/segment/predict', help='渲染结果的保存目录')
    
    args = parser.parse_args()

    # 配置计算平台与运行模式
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    # 实例化模型架构配置
    print(f"[Info] 初始化分割模型架构 (Scale: {args.scale}, Classes: {args.nc})")
    cfg = YOLOConfig(yaml_path=args.model_cfg, scale=args.scale, task="segment")
    cfg.nc = args.nc
    model = YOLO11ForSegmentation(cfg)

    # 实例化分割预测器
    predictor = SegmentationPredictor(cfg=args)

    # 执行批量推理流水线
    results = predictor(source=args.source, model=model, ckpt_path=args.weights)

    # 解析并存储预测结果
    print("\n" + "-"*50)
    for i, res in enumerate(results):
        source_item = predictor.dataset[i]
        file_name = os.path.basename(source_item) if isinstance(source_item, str) else f"pred_{i}.jpg"
        
        res.save(save_dir=args.save_dir, file_name=file_name)
        print(f"[Result] 文件 {file_name}: 检出并分割出 {len(res.det)} 个目标实例")
        
    print("-" * 50)
    print(f"[Info] 批量分割推理流水线执行完毕，结果存放于: {args.save_dir}")

if __name__ == "__main__":
    main()