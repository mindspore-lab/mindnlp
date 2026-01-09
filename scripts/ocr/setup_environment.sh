#!/bin/bash

##############################################################################
# OCR 微调环境配置脚本
# 功能：检查 NPU 环境、配置 HF 镜像、设置缓存目录
# 作者：MindNLP Team
# 日期：2026-01-06
##############################################################################

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     MindNLP OCR 微调 - 环境配置                         ║
║     NPU Environment Setup                                ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# ============================================================================
# 1. NPU 环境检查
# ============================================================================

echo ""
echo -e "${BOLD}步骤 1/3: NPU 环境检查${NC}"
echo "-----------------------------------"

# 检查 npu-smi
if command -v npu-smi &> /dev/null; then
    echo -e "${GREEN}✓ npu-smi 可用${NC}"
    npu-smi info | head -20
else
    echo -e "${YELLOW}⚠ npu-smi 未找到${NC}"
fi

# 检查 Python 包
echo ""
echo "检查 Python 依赖..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ PyTorch${NC}" || echo -e "${RED}✗ PyTorch 未安装${NC}"
python3 -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ torch_npu${NC}" || echo -e "${YELLOW}⚠ torch_npu 未安装${NC}"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ transformers${NC}" || echo -e "${RED}✗ transformers 未安装${NC}"
python3 -c "import peft; print(f'peft: {peft.__version__}')" 2>/dev/null && echo -e "${GREEN}✓ peft${NC}" || echo -e "${RED}✗ peft 未安装${NC}"
python3 -c "import datasets" 2>/dev/null && echo -e "${GREEN}✓ datasets${NC}" || echo -e "${YELLOW}⚠ datasets 未安装${NC}"

# 检查磁盘空间
echo ""
echo "检查磁盘空间..."
df -h / | tail -1
df -h /data1 2>/dev/null | tail -1 || echo "/data1 不存在"

# ============================================================================
# 2. HuggingFace 配置
# ============================================================================

echo ""
echo -e "${BOLD}步骤 2/3: HuggingFace 配置${NC}"
echo "-----------------------------------"

# 设置镜像
export HF_ENDPOINT='https://hf-mirror.com'
echo -e "${GREEN}✓ HF 镜像: ${HF_ENDPOINT}${NC}"

# Token 配置
echo ""
read -p "是否配置 HF Token（避免限流）? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "请输入你的 HF Token（或直接回车使用默认）:"
    read -p "Token: " hf_token
    
    if [ -z "${hf_token}" ]; then
        echo -e "${RED}✗ Token 未提供${NC}"
        exit 1
    else
        export HF_TOKEN="${hf_token}"
        echo -e "${GREEN}✓ Token 已设置${NC}"
    fi
    echo "  Token: ${HF_TOKEN:0:10}..."
else
    if [ -z "${HF_TOKEN}" ]; then
        echo -e "${RED}✗ 请设置 HF_TOKEN 环境变量${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 使用环境变量 Token${NC}"
fi

# ============================================================================
# 3. 缓存目录配置
# ============================================================================

echo ""
echo -e "${BOLD}步骤 3/3: 缓存目录配置${NC}"
echo "-----------------------------------"

# 检查可用空间
HOME_SPACE=$(df /home 2>/dev/null | tail -1 | awk '{print $4}')
DATA1_SPACE=$(df /data1 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")

echo "磁盘空间："
echo "  /home: $(df -h /home 2>/dev/null | tail -1 | awk '{print $4}') 可用"
if [ -d "/data1" ]; then
    echo "  /data1: $(df -h /data1 | tail -1 | awk '{print $4}') 可用"
fi

echo ""
if [ -d "/data1" ] && [ "${DATA1_SPACE}" != "0" ]; then
    read -p "使用 /data1 存储模型缓存（推荐）? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        CACHE_DIR="/data1/hf_cache"
        mkdir -p "${CACHE_DIR}"
        
        export HF_HOME="${CACHE_DIR}"
        export HF_HUB_CACHE="${CACHE_DIR}/hub"
        export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
        export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
        
        echo -e "${GREEN}✓ 缓存目录: ${CACHE_DIR}${NC}"
        echo -e "${GREEN}✓ 可用空间: $(df -h /data1 | tail -1 | awk '{print $4}')${NC}"
    else
        echo -e "${YELLOW}⚠ 使用默认缓存目录 (~/.cache)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ /data1 不可用，使用默认缓存目录${NC}"
fi

# ============================================================================
# 4. 保存配置
# ============================================================================

echo ""
echo -e "${BOLD}配置保存${NC}"
echo "-----------------------------------"

read -p "是否永久保存配置到 ~/.bashrc? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 删除旧配置
    sed -i '/# MindNLP OCR 环境配置/,/# MindNLP OCR 配置结束/d' ~/.bashrc
    
    # 添加新配置
    cat >> ~/.bashrc << EOF

# MindNLP OCR 环境配置
export HF_ENDPOINT='${HF_ENDPOINT}'
export HF_TOKEN='${HF_TOKEN}'
EOF
    
    if [ -n "${HF_HOME}" ]; then
        cat >> ~/.bashrc << EOF
export HF_HOME='${HF_HOME}'
export HF_HUB_CACHE='${HF_HUB_CACHE}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'
EOF
    fi
    
    echo "# MindNLP OCR 配置结束" >> ~/.bashrc
    
    echo -e "${GREEN}✓ 配置已保存到 ~/.bashrc${NC}"
    echo "  新终端将自动生效"
    echo "  当前终端运行: source ~/.bashrc"
else
    echo -e "${YELLOW}⚠ 配置仅在当前终端生效${NC}"
fi

# ============================================================================
# 5. 配置摘要
# ============================================================================

echo ""
echo -e "${BLUE}=========================================="
echo "配置完成！"
echo -e "==========================================${NC}"
echo ""
echo "环境变量："
echo "  HF_ENDPOINT=${HF_ENDPOINT}"
echo "  HF_TOKEN=${HF_TOKEN:0:10}..."
[ -n "${HF_HOME}" ] && echo "  HF_HOME=${HF_HOME}"
echo ""
echo "后续步骤："
echo "  1. 下载数据集: ./scripts/ocr/download_datasets.sh"
echo "  2. 转换格式: ./scripts/ocr/run_all_npu.sh (选择 3)"
echo "  3. 开始训练: ./scripts/ocr/run_all_npu.sh (选择 4)"
echo ""
echo "或使用一键菜单: ./scripts/ocr/run_all_npu.sh"
