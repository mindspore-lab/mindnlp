#!/bin/bash
# 更新服务器代码并运行测试
# 服务器: 192.168.88.19
# 在服务器上执行: bash update_server.sh

set -e  # 遇到错误立即退出

echo "🔄 更新 MindNLP OCR 代码..."
cd /data1/mindnlp

# 拉取最新代码
echo "📥 拉取最新代码..."
git fetch origin
git checkout feature/issue-2377-quantization-phase3
git pull origin feature/issue-2377-quantization-phase3

echo ""
echo "✅ 代码更新完成！"
echo ""
echo "📊 最新提交:"
git log --oneline -3
echo ""

# 显示新的目录结构
echo "📁 新的 OCR 目录结构:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lh src/mindnlp/ocr/benchmarks/ 2>/dev/null && echo "  ✓ benchmarks/" || echo "  ✗ benchmarks/ (未找到)"
ls -lh src/mindnlp/ocr/tests/ 2>/dev/null && echo "  ✓ tests/" || echo "  ✗ tests/ (未找到)"
ls -lh src/mindnlp/ocr/tools/ 2>/dev/null && echo "  ✓ tools/" || echo "  ✗ tools/ (未找到)"
ls -lh src/mindnlp/ocr/scripts/ 2>/dev/null && echo "  ✓ scripts/" || echo "  ✗ scripts/ (未找到)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 验证关键文件
echo "🔍 验证关键文件..."
files=(
    "src/mindnlp/ocr/benchmarks/benchmark_kv_cache.py"
    "src/mindnlp/ocr/benchmarks/benchmark_comparison.py"
    "src/mindnlp/ocr/tests/test_kv_cache.py"
    "src/mindnlp/ocr/models/qwen2vl.py"
    "src/mindnlp/ocr/utils/cache_manager.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
        all_exist=false
    fi
done
echo ""

if [ "$all_exist" = false ]; then
    echo "⚠️  部分文件缺失，请检查！"
    exit 1
fi

# 询问是否运行测试
read -p "🧪 是否运行快速功能测试? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🧪 运行 KV Cache 功能测试..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python src/mindnlp/ocr/tests/test_kv_cache.py || echo "⚠️  测试失败，请检查"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# 询问是否运行性能测试
read -p "📊 是否运行完整性能测试? (需要 ~30 分钟) (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📊 运行完整性能对比测试..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python src/mindnlp/ocr/benchmarks/benchmark_comparison.py \
        --model_path /data1/mindnlp_output/lora_final_20260108_222408/checkpoint-39/adapter_model.npz \
        --device npu:0 \
        --output /data1/benchmark_results/kv_cache_comparison_$(date +%Y%m%d_%H%M%S).json
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "✅ 性能测试完成！结果已保存到 /data1/benchmark_results/"
fi

echo ""
echo "✅ 服务器代码更新和测试完成！"
echo ""
echo "📝 后续操作:"
echo "  1. 查看性能测试结果: ls -lh /data1/benchmark_results/"
echo "  2. 运行单项测试: python src/mindnlp/ocr/benchmarks/benchmark_kv_cache.py --help"
echo "  3. 查看文档: cat src/mindnlp/ocr/docs/directory_structure.md"
