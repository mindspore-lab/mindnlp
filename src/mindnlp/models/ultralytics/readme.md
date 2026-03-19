# mindnlp兼容ultralutics库项目运行指南

本项目基于 MindSpore 框架实现了 YOLO11 的四大核心任务：图像分类、目标检测、实例分割和姿态估计。支持从头训练、加载预训练权重微调、模型验证与推理。

1. 数据集准备

请参考 `./ultralytics/cfg/datasets` 目录下各个数据集的配置文件（`.yaml`）获取下载路径。
下载完成后，请将数据集解压并放置到以下目录：
`./ultralytics/datasets/`

2. 预训练权重转换

我们提供了将 PyTorch 的 `.pt` 权重转化为 MindSpore 的 `.ckpt` 权重的转换脚本。请在项目根目录下运行以下命令：

```bash
# 转换分类任务权重
python tools/convert.py --task classify  
# 转换检测任务权重
python tools/convert.py --task detect
# 转换分割任务权重
python tools/convert.py --task segment
# 转换姿态估计任务权重
python tools/convert.py --task pose

3. 任务运行指令
以下命令均需在项目根目录下执行。

（1）图像分类任务 (Classify)

# 加载权重微调 (Fine-tune)
python examples/yolo/classify/run_train.py --weights ./yolo11n-cls.ckpt --save_dir ./runs/cls/train_finetune 

# 从头开始训练 (Train from scratch)
python examples/yolo/classify/run_train.py --save_dir ./runs/cls/train_scratch  

# 模型验证 (Validate)
python examples/yolo/classify/run_val.py 

# 模型推理 (Inference)
python examples/yolo/classify/inference.py
（2）目标检测任务 (Detect)

# 加载权重微调 (Fine-tune)
python examples/yolo/detect/run_train.py --weights ./yolo11n.ckpt --save_dir ./runs/detect/train_finetune 
 
# 从头开始训练 (Train from scratch)
python examples/yolo/detect/run_train.py --save_dir ./runs/detect/train_scratch 

# 模型验证 (Validate)
python examples/yolo/detect/run_val.py 
  
# 模型推理 (Inference)
python examples/yolo/detect/inference.py 
（3）实例分割任务 (Segment)

# 加载权重微调 (Fine-tune)
python examples/yolo/segment/run_train.py --weights ./yolo11n-seg.ckpt --save_dir ./runs/segment/train_finetune

# 从头开始训练 (Train from scratch)
python examples/yolo/segment/run_train.py --save_dir ./runs/segment/train_scratch

# 模型验证 (Validate)
python examples/yolo/segment/run_val.py 

# 模型推理 (Inference)
python examples/yolo/segment/inference.py 
（4）姿态估计任务 (Pose)

# 加载权重微调 (Fine-tune)
python examples/yolo/pose/run_train.py --weights ./yolo11n-pose.ckpt --save_dir ./runs/pose/train_finetune

# 从头开始训练 (Train from scratch)
python examples/yolo/pose/run_train.py --save_dir ./runs/pose/train_scratch

# 模型验证 (Validate)
python examples/yolo/pose/run_val.py 

# 模型推理 (Inference)
python examples/yolo/pose/inference.py