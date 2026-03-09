import argparse
import mindspore as ms

from models.yolo.classify.val import ClassificationValidator
from configuration_yolo import YOLOConfig
from modeling_yolo import YOLO11ForClassification

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Classification Validation Pipeline")
    
    # 核心路径配置
    parser.add_argument('--data', type=str, default='./datasets/imagenette2-160/val', help="验证集路径")
    parser.add_argument('--model_cfg', type=str, default='./cfg/models/11/yolo11-cls.yaml', help="模型结构配置文件")
    parser.add_argument('--hyp', type=str, default='./cfg/hyp.yaml', help="超参数配置文件路径")
    parser.add_argument('--weights', type=str, default='./yolo11n-cls.ckpt', help="验证所需加载的模型权重路径 (例如: best.ckpt)")
    #parser.add_argument('--weights', type=str, default='./runs/cls/train_finetune/best.ckpt', help="验证所需加载的模型权重路径 (例如: best.ckpt)")
    
    # 运行参数
    parser.add_argument('--imgsz', type=int, default=224, help="输入图像尺寸")
    parser.add_argument('--batch', type=int, default=128, help="验证批次大小")
    parser.add_argument('--workers', type=int, default=8, help="数据加载线程数")
    parser.add_argument('--device', type=str, default='Ascend', help="计算硬件类型")
    parser.add_argument('--save_dir', type=str, default="./runs/cls/val", help="验证结果保存目录")
    
    # 用于兼容基类的占位参数
    parser.add_argument('--model', type=str, default='', help="向下兼容参数")

    args = parser.parse_args()
    
    # 权重路径兼容处理
    if args.weights and not args.model:
        args.model = args.weights

    # 设置 MindSpore 运行环境
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)

    print(f"[INFO] 正在解析模型拓扑配置: {args.model_cfg}")
    cfg = YOLOConfig(yaml_path=args.model_cfg, task='classify')
    
    # 强制覆盖分类数量，实际项目中应从 args.data 对应的 dataset.yaml 中读取
    cfg.nc = getattr(args, 'nc', 10) 
    model = YOLO11ForClassification(cfg)
    
    # ---------------- 权重加载与形状对齐 ----------------
    print(f"[INFO] 正在加载检查点权重: {args.weights}")
    param_dict = ms.load_checkpoint(args.weights)
    model_dict = model.parameters_and_names()

    new_param_dict = {}
    for name, param in model_dict:
        if name in param_dict:
            ckpt_param = param_dict[name]
            
            # 权重形状匹配检查
            # 说明：当跨数据集评估时（如 ImageNet 的 1000 类权重，在 10 类的 Imagenette 上验证），
            # 最后的线性分类层尺寸将不匹配，此时忽略分类头，仅加载 Backbone 权重。
            if param.shape == ckpt_param.shape:
                new_param_dict[name] = ckpt_param
            else:
                print(f"[WARNING] 尺寸不匹配，跳过权重加载: {name} (当前模型 {param.shape} vs 检查点 {ckpt_param.shape})")
        else:
            print(f"[WARNING] 检查点中缺失权重项: {name}，当前层将保留随机初始化")

    ms.load_param_into_net(model, new_param_dict, strict_load=False)
    print(f"[INFO] 权重对齐并加载完成。")
    
    # ---------------- 验证流程启动 ----------------
    validator = ClassificationValidator(args=args)
    validator.dataloader = validator.get_dataloader(args.data, batch_size=args.batch)
    
    print(f"[INFO] DataLoader 构建完毕，即将进入评估环节...")
    
    # 启动 Validator 的 __call__ 流程
    validator(model=model)

if __name__ == "__main__":
    main()