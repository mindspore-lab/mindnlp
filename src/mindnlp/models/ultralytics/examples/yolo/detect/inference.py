import argparse
import mindspore as ms
import sys
import os

from models.yolo.detect.predict import DetectionPredictor
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForObjectDetection

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Object Detection Inference Pipeline")
    
    # --- 核心推理参数 ---
    parser.add_argument('--source', type=str, default='./datasets/coco128/coco128/images/train2017', help='待预测的图像或目录路径')
    parser.add_argument('--weights', type=str, default='./yolo11n.ckpt', help='预训练权重文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11.yaml', help='模型架构 YAML 配置文件')
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型规模标识')
    parser.add_argument('--nc', type=int, default=80, help='分类类别总数')
    
    # --- 推理后处理参数 ---
    parser.add_argument('--imgsz', type=int, default=640, help='网络输入分辨率大小')
    parser.add_argument('--conf', type=float, default=0.25, help='预测边界框的置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS 交并比阈值')
    
    # --- 运行环境参数 ---
    parser.add_argument('--device', type=str, default='Ascend', choices=['Ascend', 'GPU', 'CPU'], help='计算硬件目标')
    parser.add_argument('--save_dir', type=str, default='./runs/detect/predict', help='预测渲染结果保存目录')
    
    args = parser.parse_args()

    # 配置 MindSpore 运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    print(f"[Info] 初始化检测模型架构 (Scale: {args.scale}, Classes: {args.nc})")
    cfg = YOLOConfig(yaml_path=args.model_cfg, scale=args.scale, task="detect")
    cfg.nc = args.nc
    model = YOLO11ForObjectDetection(cfg)

    # 实例化预测器
    predictor = DetectionPredictor(cfg=args)

    # 启动推理流水线
    results = predictor(source=args.source, model=model, ckpt_path=args.weights)

    print("\n" + "-" * 50)
    for i, res in enumerate(results):
        source_item = predictor.dataset[i]
        file_name = os.path.basename(source_item) if isinstance(source_item, str) else f"pred_{i}.jpg"
        
        res.save(save_dir=args.save_dir, file_name=file_name)
        print(f"[Result] 图像 {file_name}: 共检测到 {len(res.det)} 个目标")
        
    print("-" * 50)
    print(f"[Info] 批量推理流水线执行完毕，结果存放于: {args.save_dir}")

if __name__ == "__main__":
    main()