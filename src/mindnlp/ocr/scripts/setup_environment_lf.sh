#!/bin/bash

##############################################################################
# OCR 寰皟鐜閰嶇疆鑴氭湰
# 鍔熻兘锛氭鏌?NPU 鐜銆侀厤缃?HF 闀滃儚銆佽缃紦瀛樼洰褰?# 浣滆€咃細MindNLP Team
# 鏃ユ湡锛?026-01-06
##############################################################################

set -e

# 棰滆壊瀹氫箟
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}"
cat << "EOF"
鈺斺晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?鈺?                                                          鈺?鈺?    MindNLP OCR 寰皟 - 鐜閰嶇疆                         鈺?鈺?    NPU Environment Setup                                鈺?鈺?                                                          鈺?鈺氣晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?EOF
echo -e "${NC}"

# ============================================================================
# 1. NPU 鐜妫€鏌?# ============================================================================

echo ""
echo -e "${BOLD}姝ラ 1/3: NPU 鐜妫€鏌?{NC}"
echo "-----------------------------------"

# 妫€鏌?npu-smi
if command -v npu-smi &> /dev/null; then
    echo -e "${GREEN}鉁?npu-smi 鍙敤${NC}"
    npu-smi info | head -20
else
    echo -e "${YELLOW}鈿?npu-smi 鏈壘鍒?{NC}"
fi

# 妫€鏌?Python 鍖?echo ""
echo "妫€鏌?Python 渚濊禆..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null && echo -e "${GREEN}鉁?PyTorch${NC}" || echo -e "${RED}鉁?PyTorch 鏈畨瑁?{NC}"
python3 -c "import torch_npu; print(f'torch_npu: {torch_npu.__version__}')" 2>/dev/null && echo -e "${GREEN}鉁?torch_npu${NC}" || echo -e "${YELLOW}鈿?torch_npu 鏈畨瑁?{NC}"
python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" 2>/dev/null && echo -e "${GREEN}鉁?transformers${NC}" || echo -e "${RED}鉁?transformers 鏈畨瑁?{NC}"
python3 -c "import peft; print(f'peft: {peft.__version__}')" 2>/dev/null && echo -e "${GREEN}鉁?peft${NC}" || echo -e "${RED}鉁?peft 鏈畨瑁?{NC}"
python3 -c "import datasets" 2>/dev/null && echo -e "${GREEN}鉁?datasets${NC}" || echo -e "${YELLOW}鈿?datasets 鏈畨瑁?{NC}"

# 妫€鏌ョ鐩樼┖闂?echo ""
echo "妫€鏌ョ鐩樼┖闂?.."
df -h / | tail -1
df -h /data1 2>/dev/null | tail -1 || echo "/data1 涓嶅瓨鍦?

# ============================================================================
# 2. HuggingFace 閰嶇疆
# ============================================================================

echo ""
echo -e "${BOLD}姝ラ 2/3: HuggingFace 閰嶇疆${NC}"
echo "-----------------------------------"

# 璁剧疆闀滃儚
export HF_ENDPOINT='https://hf-mirror.com'
echo -e "${GREEN}鉁?HF 闀滃儚: ${HF_ENDPOINT}${NC}"

# Token 閰嶇疆
echo ""
read -p "鏄惁閰嶇疆 HF Token锛堥伩鍏嶉檺娴侊級? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "璇疯緭鍏ヤ綘鐨?HF Token锛堟垨鐩存帴鍥炶溅浣跨敤榛樿锛?"
    read -p "Token: " hf_token
    
    if [ -z "${hf_token}" ]; then
        echo -e "${RED}鉁?Token 鏈彁渚?{NC}"
        exit 1
    else
        export HF_TOKEN="${hf_token}"
        echo -e "${GREEN}鉁?Token 宸茶缃?{NC}"
    fi
    echo "  Token: ${HF_TOKEN:0:10}..."
else
    if [ -z "${HF_TOKEN}" ]; then
        echo -e "${RED}鉁?璇疯缃?HF_TOKEN 鐜鍙橀噺${NC}"
        exit 1
    fi
    echo -e "${GREEN}鉁?浣跨敤鐜鍙橀噺 Token${NC}"
fi

# ============================================================================
# 3. 缂撳瓨鐩綍閰嶇疆
# ============================================================================

echo ""
echo -e "${BOLD}姝ラ 3/3: 缂撳瓨鐩綍閰嶇疆${NC}"
echo "-----------------------------------"

# 妫€鏌ュ彲鐢ㄧ┖闂?HOME_SPACE=$(df /home 2>/dev/null | tail -1 | awk '{print $4}')
DATA1_SPACE=$(df /data1 2>/dev/null | tail -1 | awk '{print $4}' || echo "0")

echo "纾佺洏绌洪棿锛?
echo "  /home: $(df -h /home 2>/dev/null | tail -1 | awk '{print $4}') 鍙敤"
if [ -d "/data1" ]; then
    echo "  /data1: $(df -h /data1 | tail -1 | awk '{print $4}') 鍙敤"
fi

echo ""
if [ -d "/data1" ] && [ "${DATA1_SPACE}" != "0" ]; then
    read -p "浣跨敤 /data1 瀛樺偍妯″瀷缂撳瓨锛堟帹鑽愶級? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        CACHE_DIR="/data1/hf_cache"
        mkdir -p "${CACHE_DIR}"
        
        export HF_HOME="${CACHE_DIR}"
        export HF_HUB_CACHE="${CACHE_DIR}/hub"
        export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
        export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
        
        echo -e "${GREEN}鉁?缂撳瓨鐩綍: ${CACHE_DIR}${NC}"
        echo -e "${GREEN}鉁?鍙敤绌洪棿: $(df -h /data1 | tail -1 | awk '{print $4}')${NC}"
    else
        echo -e "${YELLOW}鈿?浣跨敤榛樿缂撳瓨鐩綍 (~/.cache)${NC}"
    fi
else
    echo -e "${YELLOW}鈿?/data1 涓嶅彲鐢紝浣跨敤榛樿缂撳瓨鐩綍${NC}"
fi

# ============================================================================
# 4. 淇濆瓨閰嶇疆
# ============================================================================

echo ""
echo -e "${BOLD}閰嶇疆淇濆瓨${NC}"
echo "-----------------------------------"

read -p "鏄惁姘镐箙淇濆瓨閰嶇疆鍒?~/.bashrc? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 鍒犻櫎鏃ч厤缃?    sed -i '/# MindNLP OCR 鐜閰嶇疆/,/# MindNLP OCR 閰嶇疆缁撴潫/d' ~/.bashrc
    
    # 娣诲姞鏂伴厤缃?    cat >> ~/.bashrc << EOF

# MindNLP OCR 鐜閰嶇疆
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
    
    echo "# MindNLP OCR 閰嶇疆缁撴潫" >> ~/.bashrc
    
    echo -e "${GREEN}鉁?閰嶇疆宸蹭繚瀛樺埌 ~/.bashrc${NC}"
    echo "  鏂扮粓绔皢鑷姩鐢熸晥"
    echo "  褰撳墠缁堢杩愯: source ~/.bashrc"
else
    echo -e "${YELLOW}鈿?閰嶇疆浠呭湪褰撳墠缁堢鐢熸晥${NC}"
fi

# ============================================================================
# 5. 閰嶇疆鎽樿
# ============================================================================

echo ""
echo -e "${BLUE}=========================================="
echo "閰嶇疆瀹屾垚锛?
echo -e "==========================================${NC}"
echo ""
echo "鐜鍙橀噺锛?
echo "  HF_ENDPOINT=${HF_ENDPOINT}"
echo "  HF_TOKEN=${HF_TOKEN:0:10}..."
[ -n "${HF_HOME}" ] && echo "  HF_HOME=${HF_HOME}"
echo ""
echo "鍚庣画姝ラ锛?
echo "  1. 涓嬭浇鏁版嵁闆? ./scripts/ocr/download_datasets.sh"
echo "  2. 杞崲鏍煎紡: ./scripts/ocr/run_all_npu.sh (閫夋嫨 3)"
echo "  3. 寮€濮嬭缁? ./scripts/ocr/run_all_npu.sh (閫夋嫨 4)"
echo ""
echo "鎴栦娇鐢ㄤ竴閿彍鍗? ./scripts/ocr/run_all_npu.sh"
