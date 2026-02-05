#!/bin/bash

# OCR 并发处理功能部署脚本
# 用于在服务器上快速部署和测试

set -e

echo "=========================================="
echo "OCR 并发处理功能部署"
echo "=========================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. 检查环境
echo -e "\n${YELLOW}[1/6] 检查环境...${NC}"

# 检查 Python 版本
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3${NC}"
    exit 1
fi

python_version=$(python3 --version | cut -d ' ' -f 2)
echo -e "${GREEN}✓ Python 版本: $python_version${NC}"

# 检查 pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 pip3${NC}"
    exit 1
fi

echo -e "${GREEN}✓ pip 已安装${NC}"

# 2. 安装依赖
echo -e "\n${YELLOW}[2/6] 检查依赖...${NC}"

# 检查必要的包
required_packages=("fastapi" "uvicorn" "aiohttp" "pydantic-settings")

for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✓ $package 已安装${NC}"
    else
        echo -e "${YELLOW}安装 $package...${NC}"
        pip3 install $package
    fi
done

# 3. 配置环境变量
echo -e "\n${YELLOW}[3/6] 配置环境变量...${NC}"

# 检查是否存在 .env 文件
if [ -f "src/mindnlp/ocr/.env" ]; then
    echo -e "${GREEN}✓ .env 文件已存在${NC}"
else
    echo -e "${YELLOW}创建 .env 文件...${NC}"
    cat > src/mindnlp/ocr/.env << EOF
# OCR 服务配置

# 基础配置
OCR_USE_MOCK_ENGINE=false
OCR_DEFAULT_MODEL=Qwen/Qwen2-VL-7B-Instruct
OCR_DEVICE=npu:0

# API 配置
OCR_API_HOST=0.0.0.0
OCR_API_PORT=8000
OCR_API_WORKERS=1

# 并发处理配置 (Issue #2380)
OCR_MAX_BATCH_SIZE=4
OCR_BATCH_WAIT_TIMEOUT_MS=100
OCR_QPS_LIMIT=100
OCR_QUEUE_MAXSIZE=1000

# 日志配置
OCR_LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✓ .env 文件已创建${NC}"
fi

# 4. 验证代码
echo -e "\n${YELLOW}[4/6] 验证代码...${NC}"

# 检查核心文件是否存在
core_files=(
    "src/mindnlp/ocr/core/batching.py"
    "src/mindnlp/ocr/core/queue_limiter.py"
    "src/mindnlp/ocr/core/service_manager.py"
    "src/mindnlp/ocr/api/app.py"
)

for file in "${core_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file 不存在${NC}"
        exit 1
    fi
done

# 5. 编译检查
echo -e "\n${YELLOW}[5/6] Python 语法检查...${NC}"

for file in "${core_files[@]}"; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo -e "${GREEN}✓ $file 语法正确${NC}"
    else
        echo -e "${RED}✗ $file 语法错误${NC}"
        python3 -m py_compile "$file"
        exit 1
    fi
done

# 6. 准备测试数据
echo -e "\n${YELLOW}[6/6] 准备测试数据...${NC}"

# 创建测试目录
mkdir -p src/mindnlp/ocr/tests/test_images

# 检查测试图像
if [ -f "src/mindnlp/ocr/tests/test_images/sample.jpg" ]; then
    echo -e "${GREEN}✓ 测试图像已存在${NC}"
else
    echo -e "${YELLOW}提示: 请将测试图像放到 src/mindnlp/ocr/tests/test_images/sample.jpg${NC}"
fi

# 完成
echo -e "\n${GREEN}=========================================="
echo "部署完成！"
echo "==========================================${NC}"

echo -e "\n${YELLOW}下一步操作:${NC}"
echo -e "1. 启动服务:"
echo -e "   ${GREEN}cd src/mindnlp/ocr${NC}"
echo -e "   ${GREEN}python -m uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000${NC}"
echo -e ""
echo -e "2. 测试服务:"
echo -e "   ${GREEN}curl http://localhost:8000/api/v1/health${NC}"
echo -e ""
echo -e "3. 查看统计:"
echo -e "   ${GREEN}curl http://localhost:8000/api/v1/monitor/service-stats | jq .${NC}"
echo -e ""
echo -e "4. 运行测试:"
echo -e "   ${GREEN}cd tests${NC}"
echo -e "   ${GREEN}python test_concurrent_processing.py --base-url http://localhost:8000 --image test_images/sample.jpg --test-type concurrent${NC}"
echo -e ""

# 可选：自动启动服务
read -p "是否现在启动服务? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "\n${YELLOW}启动 OCR 服务...${NC}"
    cd src/mindnlp/ocr
    python -m uvicorn api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
fi
