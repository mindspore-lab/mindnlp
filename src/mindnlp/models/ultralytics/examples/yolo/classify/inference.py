import argparse
import os
import logging
import mindspore as ms

from models.yolo.classify.predict import ClassificationPredictor
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForClassification

# 统一日志配置
logging.basicConfig(level=logging.INFO, format='%(message)s')
LOGGER = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Classification Inference Pipeline")
    
    # --- 核心数据与权重参数 ---
    parser.add_argument('--source', type=str, default='./datasets/imagenette2-160/val/n01440764', help='待推理的图像文件或目录路径')
    parser.add_argument('--weights', type=str, default='./yolo11n-cls.ckpt', help='预训练权重文件路径')
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-cls.yaml', help='模型拓扑配置文件路径')
    parser.add_argument('--hyp', type=str, default='./cfg/hyp.yaml', help='超参数配置文件路径 (包含推理阈值配置)')
    
    # --- 架构超参数 ---
    parser.add_argument('--scale', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='模型参数规模')
    parser.add_argument('--nc', type=int, default=10, help='分类任务类别总数')
    parser.add_argument('--imgsz', type=int, default=224, help='网络输入分辨率')
    
    # --- 运行环境设定 ---
    parser.add_argument('--device', type=str, default='Ascend', help='计算硬件平台 (Ascend/GPU/CPU)')
    parser.add_argument('--save_dir', type=str, default='./runs/cls/predict', help='可视化结果持久化保存目录')
    
    args = parser.parse_args()

    # 配置 MindSpore 计算图执行模式
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    LOGGER.info(f"[INFO] 正在构建网络拓扑 (规模: {args.scale}, 类别映射空间: {args.nc})")
    cfg = YOLOConfig(yaml_path=args.model_cfg, scale=args.scale, task='classify')
    cfg.nc = args.nc
    model = YOLO11ForClassification(cfg)

    # 架构载入模块：解决迁移学习阶段的维度冲突 
    param_dict = ms.load_checkpoint(args.weights)
    model_dict = model.parameters_and_names()
    new_param_dict = {}

    for name, param in model_dict:
        if name in param_dict:
            ckpt_param = param_dict[name]
            
            # 过滤形状不匹配的参数（如分类头），防止加载权重时报错
            if param.shape == ckpt_param.shape:
                new_param_dict[name] = ckpt_param
            else:
                LOGGER.warning(f"[WARNING] 丢弃层特征权重 [{name}] | 原因: 输出维度不匹配 "
                               f"(当前网络需求: {param.shape} | 检查点参数: {ckpt_param.shape})")
        else:
            LOGGER.warning(f"[WARNING] 检查点中未匹配到网络特征层 [{name}]，已默认回退为随机初始化。")

    ms.load_param_into_net(model, new_param_dict, strict_load=False)
    LOGGER.info("[INFO] 网络主干权重数据对齐并装载完毕。")

    # 实例化推理引擎
    predictor = ClassificationPredictor(cfg=args)
    
    # 执行前向推理逻辑
    results = predictor(source=args.source, model=model, ckpt_path=None)

    # 遍历评估结果与图表保存
    LOGGER.info("\n" + "="*50)
    for i, res in enumerate(results):
        source_item = predictor.dataset[i]
        file_name = os.path.basename(source_item) if isinstance(source_item, str) else f"pred_stream_{i}.jpg"
        
        # 执行结果可视化并保存至本地文件
        res.save(save_dir=args.save_dir, file_name=file_name)
        LOGGER.info(f"成功处理图片 [{file_name}] | 预测分类: {res.top1_name:<15} | 相对置信度: {res.top1[1]:.4f}")
    
    LOGGER.info("="*50)
    absolute_save_dir = os.path.abspath(args.save_dir)
    LOGGER.info(f"[INFO] 批量推理任务执行完毕，结果文件已保存至: {absolute_save_dir}")

if __name__ == "__main__":
    main()