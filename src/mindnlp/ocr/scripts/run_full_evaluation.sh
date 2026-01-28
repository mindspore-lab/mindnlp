#!/bin/bash
# Issue #2379 完整评估流程
# 运行基线和LoRA模型评估，生成最终验收报告

set -e

# 🔧 将所有缓存和临时文件重定向到 /data1 (避免根目录磁盘满)
export TMPDIR=/data1/tmp
export TEMP=/data1/tmp
export TMP=/data1/tmp

# Ascend NPU 日志和缓存
export ASCEND_PROCESS_LOG_PATH=/data1/ascend_logs
export ASCEND_GLOBAL_LOG_PATH=/data1/ascend_logs
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# Python/Hugging Face/Torch 缓存
export HF_HOME=/data1/.cache/huggingface
export TRANSFORMERS_CACHE=/data1/.cache/huggingface/transformers
export HF_DATASETS_CACHE=/data1/.cache/huggingface/datasets
export TORCH_HOME=/data1/.cache/torch
export XDG_CACHE_HOME=/data1/.cache

# NPU Kernel 编译缓存
export KERNEL_META_CACHE=/data1/.cache/kernel_meta
mkdir -p "$KERNEL_META_CACHE"
ln -sf "$KERNEL_META_CACHE" "$HOME/mindnlp/kernel_meta" 2>/dev/null || true

# 创建必要的目录
mkdir -p /data1/tmp
mkdir -p /data1/ascend_logs
mkdir -p /data1/.cache/huggingface
mkdir -p /data1/.cache/torch

# 清理旧文件
rm -rf /data1/tmp/tmp* 2>/dev/null || true
rm -rf /data1/ascend_logs/plog/* 2>/dev/null || true

# 清理根目录下的 Ascend 日志和缓存(释放空间)
echo "🧹 清理磁盘空间..."
rm -rf /home/$USER/ascend/log/run/plog/* 2>/dev/null || true
rm -rf /home/$USER/Ascend/latest/log/* 2>/dev/null || true

# 创建 kernel_meta 缓存目录并设置符号链接
echo "🔗 设置 kernel_meta 符号链接..."
mkdir -p "$KERNEL_META_CACHE"
# 删除旧的 kernel_meta (可能是目录或符号链接)
rm -rf "$HOME/mindnlp/kernel_meta"
# 创建新的符号链接
ln -sf "$KERNEL_META_CACHE" "$HOME/mindnlp/kernel_meta"
echo "   kernel_meta -> $KERNEL_META_CACHE"

# 清理其他旧缓存
rm -rf ~/.cache/huggingface 2>/dev/null || true
rm -rf ~/.cache/torch 2>/dev/null || true

df -h / | grep -v Filesystem
echo ""

echo "========================================"
echo "Issue #2379 完整评估流程"
echo "========================================"
echo ""

# 配置
BASE_MODEL="/data1/models/qwen2vl_7b_merged"
LORA_PATH="/data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39"
TEST_DATA="/data1/ocr_test"
OUTPUT_DIR="/data1/evaluation_results"
MINDNLP_ROOT="$HOME/mindnlp"
BATCH_SIZE=8  # 批量大小,可根据显存调整(NPU 34GB可以用8-16)

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "📊 配置信息:"
echo "  基础模型: $BASE_MODEL"
echo "  LoRA模型: $LORA_PATH"
echo "  测试数据: $TEST_DATA"
echo "  输出目录: $OUTPUT_DIR"
echo "  批量大小: $BATCH_SIZE"
echo "  临时目录: $TMPDIR"
echo ""

# 检查数据集
echo "🔍 检查数据集..."
for dataset in icdar2015 funsd sroie; do
    if [ -d "$TEST_DATA/$dataset" ]; then
        count=$(ls "$TEST_DATA/$dataset"/*.json 2>/dev/null | wc -l)
        echo "  ✅ $dataset: $count 文件"
    else
        echo "  ❌ $dataset: 未找到"
    fi
done
echo ""

# 检查LoRA模型
echo "🔍 检查LoRA模型..."
if [ -d "$LORA_PATH" ]; then
    echo "  ✅ LoRA checkpoint存在"
    ls -lh "$LORA_PATH"
else
    echo "  ❌ LoRA checkpoint不存在: $LORA_PATH"
    exit 1
fi
echo ""

cd "$MINDNLP_ROOT"

# 步骤1: 评估LoRA模型
echo "========================================"
echo "步骤 1/3: 评估LoRA模型"
echo "========================================"
echo ""

# 评估ICDAR 2015数据集
echo "📊 评估ICDAR 2015数据集..."
python3 src/mindnlp/ocr/finetune/evaluate.py \
    --model_path "$LORA_PATH" \
    --base_model_path "$BASE_MODEL" \
    --test_data_path "$TEST_DATA/icdar2015/test.json" \
    --output_file "$OUTPUT_DIR/lora_icdar2015_results.json" \
    --batch_size $BATCH_SIZE

# 评估FUNSD数据集
echo ""
echo "📊 评估FUNSD数据集..."
python3 src/mindnlp/ocr/finetune/evaluate.py \
    --model_path "$LORA_PATH" \
    --base_model_path "$BASE_MODEL" \
    --test_data_path "$TEST_DATA/funsd/test.json" \
    --output_file "$OUTPUT_DIR/lora_funsd_results.json" \
    --batch_size $BATCH_SIZE

# 评估SROIE数据集
echo ""
echo "📊 评估SROIE数据集..."
python3 src/mindnlp/ocr/finetune/evaluate.py \
    --model_path "$LORA_PATH" \
    --base_model_path "$BASE_MODEL" \
    --test_data_path "$TEST_DATA/sroie/test.json" \
    --output_file "$OUTPUT_DIR/lora_sroie_results.json" \
    --batch_size $BATCH_SIZE

# 合并评估结果
echo ""
echo "📊 合并评估结果..."
python3 -c "
import json
results = []
for dataset in ['icdar2015', 'funsd', 'sroie']:
    file = '$OUTPUT_DIR/lora_{}_results.json'.format(dataset)
    try:
        with open(file) as f:
            data = json.load(f)
            results.append({
                'dataset': dataset,
                'metrics': data
            })
    except FileNotFoundError:
        print(f'Warning: {file} not found')
        
with open('$OUTPUT_DIR/lora_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('✅ 结果已合并到 lora_results.json')
"

echo ""
echo "✅ LoRA模型评估完成"
echo ""

# 步骤2: 生成验收报告
echo "========================================"
echo "步骤 2/3: 生成验收报告"
echo "========================================"
echo ""

python3 scripts/ocr/validate_acceptance_criteria.py \
    --mode all \
    --lora_results "$OUTPUT_DIR/lora_results.json" \
    --lora_path "$LORA_PATH"

echo ""
echo "✅ 验收报告生成完成"
echo ""

# 步骤3: 显示结果摘要
echo "========================================"
echo "步骤 3/3: 结果摘要"
echo "========================================"
echo ""

if [ -f "acceptance_report.json" ]; then
    echo "📊 验收报告内容:"
    python3 -c "
import json
with open('acceptance_report.json') as f:
    data = json.load(f)
    print(f\"总体结果: {data['summary']['passed']}/{data['summary']['total']} 项达标 ({data['summary']['pass_rate']:.1f}%)\")
    print()
    for criteria in data['criteria']:
        status = '✅' if criteria['passed'] else '❌'
        print(f\"{status} {criteria['description']}\")
"
fi

echo ""
echo "========================================"
echo "✅ 完整评估流程结束"
echo "========================================"
echo ""
echo "结果文件位置:"
echo "  - LoRA评估结果: $OUTPUT_DIR/lora_results.json"
echo "  - 验收报告: $MINDNLP_ROOT/acceptance_report.json"
echo ""
