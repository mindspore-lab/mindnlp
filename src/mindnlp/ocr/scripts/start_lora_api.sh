#!/bin/bash

# 启动使用LoRA微调模型的OCR API服务
# 使用方法: bash start_lora_api.sh

# 设置环境变量
export TMPDIR=/data1/tmp
export HF_HOME=/data1/huggingface_cache
export TRANSFORMERS_CACHE=/data1/huggingface_cache

# LoRA模型路径 - 修改为你的实际路径
export OCR_LORA_WEIGHTS_PATH="/data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39"

# API配置
export OCR_API_HOST="0.0.0.0"
export OCR_API_PORT=8000
export OCR_DEVICE="npu:0"
export OCR_DEFAULT_MODEL="Qwen/Qwen2-VL-7B-Instruct"

# 日志级别
export OCR_LOG_LEVEL="INFO"

# 镜像站点
export HF_ENDPOINT='https://hf-mirror.com'
export HF_HUB_ENDPOINT='https://hf-mirror.com'

# 离线模式（模型已下载）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=========================================="
echo "启动 LoRA 微调模型 API 服务"
echo "=========================================="
echo "基础模型: $OCR_DEFAULT_MODEL"
echo "LoRA权重: $OCR_LORA_WEIGHTS_PATH"
echo "设备: $OCR_DEVICE"
echo "端口: $OCR_API_PORT"
echo "=========================================="
echo ""
# 调试：确认环境变量已设置
echo "🔍 环境变量检查:"
echo "  OCR_LORA_WEIGHTS_PATH=${OCR_LORA_WEIGHTS_PATH}"
echo ""
# 启动服务
cd ~/mindnlp
python src/mindnlp/ocr/main.py
