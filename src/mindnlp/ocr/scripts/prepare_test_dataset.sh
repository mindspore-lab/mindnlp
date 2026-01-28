#!/bin/bash
# OCR测试数据集准备脚本
# 用于下载和转换公开数据集

set -e

# 配置
DATASET_ROOT="${DATASET_ROOT:-/data1/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/data1/ocr_test}"
PYTHON="${PYTHON:-python}"

echo "=========================================="
echo "OCR Test Dataset Preparation"
echo "=========================================="
echo "Dataset root: $DATASET_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo ""

# 创建目录
mkdir -p "$DATASET_ROOT"
mkdir -p "$OUTPUT_DIR"

# 函数：转换ICDAR 2015数据集
prepare_icdar2015() {
    echo "📦 Preparing ICDAR 2015 dataset..."
    
    ICDAR_DIR="$DATASET_ROOT/icdar2015"
    
    if [ ! -d "$ICDAR_DIR" ]; then
        echo "⚠️  ICDAR 2015 数据集未找到"
        echo "请从以下地址下载数据集："
        echo "  https://rrc.cvc.uab.es/?ch=4&com=downloads"
        echo "并解压到: $ICDAR_DIR"
        return 1
    fi
    
    # 转换训练集
    if [ -d "$ICDAR_DIR/train" ]; then
        $PYTHON -m mindnlp.ocr.finetune.prepare_dataset \
            --format icdar2015 \
            --data_dir "$ICDAR_DIR" \
            --output_dir "$OUTPUT_DIR/icdar2015" \
            --split train \
            --validate
        echo "✅ ICDAR 2015 训练集转换完成"
    fi
    
    # 转换测试集
    if [ -d "$ICDAR_DIR/test" ]; then
        $PYTHON -m mindnlp.ocr.finetune.prepare_dataset \
            --format icdar2015 \
            --data_dir "$ICDAR_DIR" \
            --output_dir "$OUTPUT_DIR/icdar2015" \
            --split test \
            --validate
        echo "✅ ICDAR 2015 测试集转换完成"
    fi
}

# 函数：转换FUNSD数据集
prepare_funsd() {
    echo "📦 Preparing FUNSD dataset..."
    
    FUNSD_DIR="$DATASET_ROOT/funsd"
    
    if [ ! -d "$FUNSD_DIR" ]; then
        echo "⚠️  FUNSD 数据集未找到"
        echo "请从以下地址下载数据集:"
        echo "  https://guillaumejaume.github.io/FUNSD/"
        echo "并解压到: $FUNSD_DIR"
        return 1
    fi
    
    # 转换训练集
    if [ -d "$FUNSD_DIR/train" ]; then
        $PYTHON -m mindnlp.ocr.finetune.prepare_dataset \
            --format funsd \
            --data_dir "$FUNSD_DIR" \
            --output_dir "$OUTPUT_DIR/funsd" \
            --split train \
            --validate
        echo "✅ FUNSD 训练集转换完成"
    fi
    
    # 转换测试集
    if [ -d "$FUNSD_DIR/test" ]; then
        $PYTHON -m mindnlp.ocr.finetune.prepare_dataset \
            --format funsd \
            --data_dir "$FUNSD_DIR" \
            --output_dir "$OUTPUT_DIR/funsd" \
            --split test \
            --validate
        echo "✅ FUNSD 测试集转换完成"
    fi
}

# 函数：转换SROIE数据集
prepare_sroie() {
    echo "📦 Preparing SROIE dataset..."
    
    SROIE_DIR="$DATASET_ROOT/sroie"
    
    if [ ! -d "$SROIE_DIR" ]; then
        echo "⚠️  SROIE 数据集未找到"
        echo "请从以下地址下载数据集:"
        echo "  https://rrc.cvc.uab.es/?ch=13&com=downloads"
        echo "并解压到: $SROIE_DIR"
        return 1
    fi
    
    # 转换训练集
    if [ -d "$SROIE_DIR/train" ]; then
        $PYTHON -m mindnlp.ocr.finetune.prepare_dataset \
            --format sroie \
            --data_dir "$SROIE_DIR" \
            --output_dir "$OUTPUT_DIR/sroie" \
            --split train \
            --validate
        echo "✅ SROIE 训练集转换完成"
    fi
    
    # 转换测试集
    if [ -d "$SROIE_DIR/test" ]; then
        $PYTHON -m mindnlp.ocr.finetune.prepare_dataset \
            --format sroie \
            --data_dir "$SROIE_DIR" \
            --output_dir "$OUTPUT_DIR/sroie" \
            --split test \
            --validate
        echo "✅ SROIE 测试集转换完成"
    fi
}

# 主流程
case "${1:-all}" in
    icdar|icdar2015)
        prepare_icdar2015
        ;;
    funsd)
        prepare_funsd
        ;;
    sroie)
        prepare_sroie
        ;;
    all)
        prepare_icdar2015 || echo "⚠️  ICDAR 2015 跳过"
        prepare_funsd || echo "⚠️  FUNSD 跳过"
        prepare_sroie || echo "⚠️  SROIE 跳过"
        ;;
    *)
        echo "Usage: $0 [icdar|funsd|sroie|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Dataset preparation completed!"
echo "=========================================="
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "下一步："
echo "1. 检查转换后的数据集"
echo "2. 运行训练脚本进行微调"
echo "3. 使用评估脚本验证模型性能"
