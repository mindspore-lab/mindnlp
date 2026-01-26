#!/bin/bash
# 部署并发处理模块到服务器

echo "=========================================="
echo "部署并发处理优化模块"
echo "=========================================="

SERVER="mseco@192.168.88.19"
BASE_PATH="~/mindnlp/src/mindnlp/ocr"

echo ""
echo "步骤 1: 创建必要的目录..."
ssh $SERVER "mkdir -p ${BASE_PATH}/core/batch"
ssh $SERVER "mkdir -p ${BASE_PATH}/core/limiter"
ssh $SERVER "mkdir -p ${BASE_PATH}/core/metrics"
ssh $SERVER "mkdir -p ${BASE_PATH}/core/concurrency"
ssh $SERVER "mkdir -p ${BASE_PATH}/api/routes"

echo ""
echo "步骤 2: 传输批处理模块..."
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/batch/dynamic_batcher.py" "$SERVER:${BASE_PATH}/core/batch/"
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/batch/__init__.py" "$SERVER:${BASE_PATH}/core/batch/"

echo ""
echo "步骤 3: 传输限流器模块..."
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/limiter/rate_limiter.py" "$SERVER:${BASE_PATH}/core/limiter/"
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/limiter/__init__.py" "$SERVER:${BASE_PATH}/core/limiter/"

echo ""
echo "步骤 4: 传输监控模块..."
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/metrics/performance.py" "$SERVER:${BASE_PATH}/core/metrics/"
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/metrics/__init__.py" "$SERVER:${BASE_PATH}/core/metrics/"

echo ""
echo "步骤 5: 传输并发管理器..."
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/concurrency/manager.py" "$SERVER:${BASE_PATH}/core/concurrency/"
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/core/concurrency/__init__.py" "$SERVER:${BASE_PATH}/core/concurrency/"

echo ""
echo "步骤 6: 传输监控API路由..."
scp "d:/开源实习/mindnlp/src/mindnlp/ocr/api/routes/metrics.py" "$SERVER:${BASE_PATH}/api/routes/"

echo ""
echo "步骤 7: 传输测试脚本..."
scp "d:/开源实习/mindnlp/tests/load_test.py" "$SERVER:~/mindnlp/tests/"

echo ""
echo "步骤 8: 清除Python缓存..."
ssh $SERVER "cd ~/mindnlp && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true"

echo ""
echo "=========================================="
echo "✅ 部署完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 重启OCR服务"
echo "2. 运行性能测试"
echo ""
