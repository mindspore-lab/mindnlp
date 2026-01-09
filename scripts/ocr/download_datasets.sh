#!/bin/bash

##############################################################################
# OCR 数据集下载脚本
# 支持下载: ICDAR 2015, SROIE, FUNSD, DocVQA
# 作者：MindNLP Team
# 日期：2026-01-06
##############################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================="
echo "OCR 数据集下载工具"
echo -e "==========================================${NC}"

# ============================================================================
# 配置
# ============================================================================

# 基础路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# 数据集目录
DATASET_ROOT="${DATASET_ROOT:-./datasets}"
mkdir -p "${DATASET_ROOT}"

# Hugging Face 镜像配置
# 如果无法访问 huggingface.co，使用国内镜像
if [ -z "${HF_ENDPOINT}" ]; then
    echo -e "${YELLOW}提示: 如果下载失败，将自动切换到 HF Mirror 镜像${NC}"
    export HF_ENDPOINT="https://hf-mirror.com"
fi

# HF Token 配置（避免限流）
if [ -z "${HF_TOKEN}" ]; then
    echo -e "${YELLOW}提示: 未设置 HF_TOKEN，可能遇到限流问题${NC}"
    echo -e "${YELLOW}可以设置: export HF_TOKEN='your_token_here'${NC}"
fi

echo ""
echo -e "${GREEN}数据集将下载到: ${DATASET_ROOT}${NC}"
echo -e "${GREEN}HF 镜像地址: ${HF_ENDPOINT}${NC}"
if [ -n "${HF_TOKEN}" ]; then
    echo -e "${GREEN}HF Token: 已设置 (${HF_TOKEN:0:10}...)${NC}"
else
    echo -e "${YELLOW}HF Token: 未设置${NC}"
fi
echo ""

# ============================================================================
# 辅助函数
# ============================================================================

download_with_wget() {
    local url=$1
    local output=$2
    echo "下载: ${url}"
    wget -c "${url}" -O "${output}" || {
        echo -e "${RED}下载失败: ${url}${NC}"
        return 1
    }
}

download_with_curl() {
    local url=$1
    local output=$2
    echo "下载: ${url}"
    curl -L -C - "${url}" -o "${output}" || {
        echo -e "${RED}下载失败: ${url}${NC}"
        return 1
    }
}

extract_archive() {
    local file=$1
    local dest=$2
    echo "解压: ${file}"
    
    if [[ "${file}" == *.zip ]]; then
        unzip -q "${file}" -d "${dest}"
    elif [[ "${file}" == *.tar.gz ]] || [[ "${file}" == *.tgz ]]; then
        tar -xzf "${file}" -C "${dest}"
    elif [[ "${file}" == *.tar ]]; then
        tar -xf "${file}" -C "${dest}"
    else
        echo -e "${YELLOW}⚠ 未知的压缩格式: ${file}${NC}"
        return 1
    fi
}

# ============================================================================
# 1. ICDAR 2015 Dataset (场景文本检测与识别)
# ============================================================================

download_icdar2015() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "下载 ICDAR 2015 Dataset"
    echo -e "==========================================${NC}"
    
    local dataset_dir="${DATASET_ROOT}/icdar2015"
    mkdir -p "${dataset_dir}"
    
    echo ""
    echo "ICDAR 2015 需要在官网注册后手动下载"
    echo "官网: https://rrc.cvc.uab.es/?ch=4"
    echo ""
    echo "下载后请将文件放置到: ${dataset_dir}"
    echo ""
    echo "所需文件:"
    echo "  - ch4_training_images.zip"
    echo "  - ch4_training_localization_transcription_gt.zip"
    echo "  - ch4_test_images.zip"
    echo ""
    
    read -p "是否已下载并准备就绪? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "请先下载数据集"
        return 1
    fi
    
    # 检查文件
    if [ -f "${dataset_dir}/ch4_training_images.zip" ]; then
        echo "解压训练图片..."
        extract_archive "${dataset_dir}/ch4_training_images.zip" "${dataset_dir}"
    fi
    
    if [ -f "${dataset_dir}/ch4_training_localization_transcription_gt.zip" ]; then
        echo "解压训练标注..."
        extract_archive "${dataset_dir}/ch4_training_localization_transcription_gt.zip" "${dataset_dir}"
    fi
    
    echo -e "${GREEN}✓ ICDAR 2015 准备完成${NC}"
    echo "数据位置: ${dataset_dir}"
}

# ============================================================================
# 2. SROIE Dataset (收据文字提取)
# ============================================================================

download_sroie() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "下载 SROIE Dataset"
    echo -e "==========================================${NC}"
    
    local dataset_dir="${DATASET_ROOT}/sroie"
    mkdir -p "${dataset_dir}"
    
    echo ""
    echo "SROIE 使用 huggingface-cli 下载"
    echo ""
    
    # 检查是否安装 huggingface-cli
    if ! command -v huggingface-cli &> /dev/null; then
        echo -e "${YELLOW}⚠ huggingface-cli 未安装${NC}"
        read -p "是否安装? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            pip install -U huggingface-hub
        else
            echo "跳过 SROIE 下载"
            return 0
        fi
    fi
    
    # 使用 huggingface-cli 下载
    echo "下载 SROIE 数据集..."
    export HF_ENDPOINT="${HF_ENDPOINT}"
    
    if [ -n "${HF_TOKEN}" ]; then
        huggingface-cli download darentang/sroie \
            --repo-type dataset \
            --local-dir "${dataset_dir}" \
            --token "${HF_TOKEN}"
    else
        huggingface-cli download darentang/sroie \
            --repo-type dataset \
            --local-dir "${dataset_dir}"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ SROIE 下载完成${NC}"
        echo "数据位置: ${dataset_dir}"
        echo ""
        echo "注意: 此数据集需要手动转换为 MindNLP 格式"
        echo "运行: python3 scripts/ocr/convert_datasets.py sroie ${dataset_dir} ${dataset_dir}_converted"
    else
        echo -e "${RED}✗ SROIE 下载失败${NC}"
        echo ""
        echo "建议使用其他数据集:"
        echo "  - FUNSD (表单理解)"
        echo "  - DocVQA (文档问答)"
        return 1
    fi
}

# ============================================================================
# 3. FUNSD Dataset (表单理解)
# ============================================================================

download_funsd() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "下载 FUNSD Dataset"
    echo -e "==========================================${NC}"
    
    local dataset_dir="${DATASET_ROOT}/funsd"
    mkdir -p "${dataset_dir}"
    
    echo ""
    echo "从 Hugging Face 下载 FUNSD..."
    echo ""
    
    # 使用 Python 下载
    export HF_ENDPOINT="${HF_ENDPOINT}"
    export HF_TOKEN="${HF_TOKEN}"
    python << 'EOF'
from datasets import load_dataset
import os

print("正在下载 FUNSD 数据集...")
print(f"使用镜像: {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}")

# 获取 token
token = os.getenv('HF_TOKEN')
if token:
    print(f"使用 Token: {token[:10]}...")

try:
    dataset = load_dataset("nielsr/funsd", token=token)
    
    # 保存到本地
    dataset_dir = os.getenv('DATASET_ROOT', './datasets') + '/funsd'
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"保存数据集到: {dataset_dir}")
    dataset.save_to_disk(dataset_dir)
    
    print("✓ FUNSD 下载完成")
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"测试集样本数: {len(dataset['test'])}")
except Exception as e:
    print(f"下载失败: {e}")
    print("\n建议:")
    print("1. 使用 huggingface-cli 手动下载:")
    print("   huggingface-cli download nielsr/funsd --repo-type dataset --local-dir ./datasets/funsd")
    exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ FUNSD 下载完成${NC}"
        echo "数据位置: ${dataset_dir}"
    else
        echo -e "${RED}✗ FUNSD 下载失败${NC}"
        return 1
    fi
}

# ============================================================================
# 4. DocVQA Dataset (文档视觉问答)
# ============================================================================

download_docvqa() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "下载 DocVQA Dataset"
    echo -e "==========================================${NC}"
    
    local dataset_dir="${DATASET_ROOT}/docvqa"
    mkdir -p "${dataset_dir}"
    
    echo ""
    echo "DocVQA 需要从官网或 Hugging Face 下载"
    echo "官网: https://rrc.cvc.uab.es/?ch=17"
    echo "Hugging Face: https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut"
    echo ""
    
    read -p "使用 Hugging Face 小样本版本 (1200 examples)? (Y/n) " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        # 下载小样本版本
        export HF_ENDPOINT="${HF_ENDPOINT}"
        export HF_TOKEN="${HF_TOKEN}"
        python << 'EOF'
from datasets import load_dataset
import os

print("正在下载 DocVQA 小样本数据集...")
print(f"使用镜像: {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}")

# 获取 token
token = os.getenv('HF_TOKEN')
if token:
    print(f"使用 Token: {token[:10]}...")

try:
    dataset = load_dataset("nielsr/docvqa_1200_examples_donut", token=token)
    
    # 保存到本地
    dataset_dir = os.getenv('DATASET_ROOT', './datasets') + '/docvqa'
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"保存数据集到: {dataset_dir}")
    dataset.save_to_disk(dataset_dir)
    
    print("✓ DocVQA 下载完成")
    for split in dataset.keys():
        print(f"{split} 样本数: {len(dataset[split])}")
except Exception as e:
    print(f"下载失败: {e}")
    print("\n建议:")
    print("1. 使用 huggingface-cli 手动下载:")
    print("   huggingface-cli download nielsr/docvqa_1200_examples_donut --repo-type dataset --local-dir ./datasets/docvqa")
    exit(1)
EOF
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ DocVQA 下载完成${NC}"
            echo "数据位置: ${dataset_dir}"
        else
            echo -e "${RED}✗ DocVQA 下载失败${NC}"
            return 1
        fi
    else
        echo "请从官网下载完整版数据集"
        echo "下载后解压到: ${dataset_dir}"
    fi
}

# ============================================================================
# 数据转换为 MindNLP 格式
# ============================================================================

convert_to_mindnlp_format() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "转换数据集为 MindNLP OCR 格式"
    echo -e "==========================================${NC}"
    echo ""
    
    read -p "选择要转换的数据集 (1:SROIE 2:FUNSD 3:DocVQA): " dataset_choice
    
    case $dataset_choice in
        1)
            echo "转换 SROIE..."
            python << 'EOF'
import json
import os
from datasets import load_from_disk

dataset_dir = os.getenv('DATASET_ROOT', './datasets') + '/sroie'
output_file = dataset_dir + '/train_mindnlp.json'

print(f"加载数据集: {dataset_dir}")
dataset = load_from_disk(dataset_dir)

samples = []
for idx, item in enumerate(dataset['train']):
    # SROIE 格式转换
    sample = {
        "image_path": f"images/{idx}.jpg",  # 需要手动保存图片
        "conversations": [
            {
                "role": "user",
                "content": "请识别这张收据中的文字"
            },
            {
                "role": "assistant",
                "content": item.get('text', '')  # 根据实际字段调整
            }
        ],
        "task_type": "receipt"
    }
    samples.append(sample)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)

print(f"✓ 已保存到: {output_file}")
print(f"样本数: {len(samples)}")
EOF
            ;;
        2)
            echo "转换 FUNSD..."
            echo "FUNSD 是表单理解数据集，需要自定义转换逻辑"
            ;;
        3)
            echo "转换 DocVQA..."
            echo "DocVQA 是问答数据集，需要自定义转换逻辑"
            ;;
        *)
            echo "无效选择"
            ;;
    esac
}

# ============================================================================
# 显示菜单
# ============================================================================

show_menu() {
    echo ""
    echo -e "${BLUE}=========================================="
    echo "可下载的数据集:"
    echo -e "==========================================${NC}"
    echo "1. ICDAR 2015 (场景文本检测)"
    echo "2. SROIE (收据文字提取)"
    echo "3. FUNSD (表单理解)"
    echo "4. DocVQA (文档问答)"
    echo "5. 全部下载"
    echo "6. 转换为 MindNLP 格式"
    echo "0. 退出"
    echo "-----------------------------------"
    echo ""
}

# ============================================================================
# 主函数
# ============================================================================

main() {
    # 检查依赖
    echo "检查依赖..."
    
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        echo -e "${RED}✗ 需要安装 wget 或 curl${NC}"
        exit 1
    fi
    
    if ! command -v unzip &> /dev/null; then
        echo -e "${YELLOW}⚠ 建议安装 unzip${NC}"
    fi
    
    if ! python -c "import datasets" &> /dev/null; then
        echo -e "${YELLOW}⚠ 建议安装 datasets: pip install datasets${NC}"
    fi
    
    echo -e "${GREEN}✓ 依赖检查完成${NC}"
    
    # 显示菜单
    while true; do
        show_menu
        read -p "请选择 [0-6]: " choice
        
        case $choice in
            1)
                download_icdar2015
                ;;
            2)
                download_sroie
                ;;
            3)
                download_funsd
                ;;
            4)
                download_docvqa
                ;;
            5)
                echo ""
                echo "开始下载所有数据集..."
                download_sroie
                download_funsd
                download_docvqa
                echo ""
                echo -e "${YELLOW}提示: ICDAR 2015 需要手动下载${NC}"
                ;;
            6)
                convert_to_mindnlp_format
                ;;
            0)
                echo ""
                echo "退出"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                ;;
        esac
        
        echo ""
        read -p "按回车键继续..."
    done
}

# 运行主程序
main
