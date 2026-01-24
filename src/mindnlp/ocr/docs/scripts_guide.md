# OCR Toolkit 使用指南

统一的 OCR 训练、推理和评估工具。

## 文件说明

### 核心工具
- **ocr_toolkit.py** - 主工具脚本，包含训练、推理、评估三大功能

### 数据准备
- **download_datasets.sh** - 下载 FUNSD 数据集
- **convert_datasets.py** - 转换数据集格式
- **convert_features_to_numpy.py** - 提取视觉特征到 NumPy

### 环境配置
- **setup_environment.sh** / **setup_environment_lf.sh** - 环境配置脚本

## 使用方法

### 1. 训练模型

```bash
python ocr_toolkit.py train \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --data_path ./datasets/funsd_converted/train.json \
    --features_dir ./datasets/funsd_converted/visual_features_numpy \
    --output_dir /data1/output/lora_$(date +%Y%m%d_%H%M%S) \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --max_length 256 \
    --device npu:0
```

### 2. 推理

#### 单张图片推理
```bash
python ocr_toolkit.py infer \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --checkpoint_dir /data1/output/lora_xxx/checkpoint-39 \
    --image_path test.jpg \
    --device npu:0
```

#### 批量推理
```bash
python ocr_toolkit.py infer \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --checkpoint_dir /data1/output/lora_xxx/checkpoint-39 \
    --test_data_path ./datasets/funsd_converted/test.json \
    --output_file predictions.json \
    --device npu:0
```

### 3. 评估

```bash
python ocr_toolkit.py eval \
    --predictions_file predictions.json \
    --output_file evaluation_results.json
```

## 完整流程

### 1. 准备数据
```bash
# 下载数据集
bash download_datasets.sh

# 转换格式
python convert_datasets.py \
    --funsd_dir ./datasets/funsd \
    --output_dir ./datasets/funsd_converted

# 提取视觉特征
python convert_features_to_numpy.py \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --data_path ./datasets/funsd_converted/train.json \
    --output_dir ./datasets/funsd_converted/visual_features_numpy \
    --device npu:0
```

### 2. 训练模型
```bash
python ocr_toolkit.py train \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --data_path ./datasets/funsd_converted/train.json \
    --features_dir ./datasets/funsd_converted/visual_features_numpy \
    --output_dir /data1/output/lora_final \
    --device npu:0
```

### 3. 测试模型
```bash
# 批量推理
python ocr_toolkit.py infer \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --checkpoint_dir /data1/output/lora_final/checkpoint-39 \
    --test_data_path ./datasets/funsd_converted/test.json \
    --output_file predictions.json \
    --device npu:0

# 评估
python ocr_toolkit.py eval \
    --predictions_file predictions.json \
    --output_file evaluation.json
```

## 已训练模型

**位置**: `/data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39/`

**配置**:
- LoRA r=8, alpha=16, dropout=0.1
- 训练: 3 epochs, 39 steps
- Loss: 3.4267 → 3.381
- 大小: 15.52 GB (1122个张量)

**使用示例**:
```bash
python ocr_toolkit.py infer \
    --model_path Qwen/Qwen2-VL-7B-Instruct \
    --checkpoint_dir /data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39 \
    --image_path your_image.jpg \
    --device npu:0
```

## 注意事项

1. **磁盘空间**: 确保输出目录有足够空间（至少20GB）
2. **临时目录**: 设置 `export TMPDIR=/data1/tmp` 避免磁盘满
3. **离线模式**: 如遇网络问题，设置 `export HF_HUB_OFFLINE=1`
4. **设备选择**: NPU使用 `npu:0`，GPU使用 `cuda:0`，CPU使用 `cpu`

## 故障排除

### 磁盘空间不足
```bash
# 清理日志
rm -rf ~/ascend/log/*

# 使用 /data1 作为输出目录
--output_dir /data1/output/...
```

### 网络问题
```bash
# 使用离线模式（模型已缓存）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### NPU 编译缓存问题
```bash
# 清理 NPU 缓存
rm -rf ~/.cache/Ascend
```
