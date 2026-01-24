#!/bin/bash
# 自动整理 OCR 文件结构
# 在服务器上执行: bash reorganize_files.sh

set -e

echo "🔧 开始整理 OCR 文件结构..."
cd /data1/mindnlp

# 1. 创建新目录
echo ""
echo "📁 创建新目录结构..."
mkdir -p src/mindnlp/ocr/{benchmarks,tests,tools,scripts,docs}

# 2. 检查旧位置的文件
echo ""
echo "🔍 检查 scripts/ocr/ 下的文件..."
if [ -d "scripts/ocr" ]; then
    ls -lh scripts/ocr/ | grep -v "^total" | grep -v "^d"
    echo ""
else
    echo "⚠️  scripts/ocr/ 目录不存在"
    exit 1
fi

# 3. 移动 benchmark 文件
echo "📦 移动 benchmark 文件..."
if ls scripts/ocr/benchmark_*.py 1> /dev/null 2>&1; then
    mv scripts/ocr/benchmark_*.py src/mindnlp/ocr/benchmarks/
    echo "  ✓ benchmark_*.py → benchmarks/"
fi

if [ -f "scripts/ocr/validate_acceptance_criteria.py" ]; then
    mv scripts/ocr/validate_acceptance_criteria.py src/mindnlp/ocr/benchmarks/
    echo "  ✓ validate_acceptance_criteria.py → benchmarks/"
fi

# 4. 移动测试文件
echo ""
echo "📦 移动测试文件..."
if ls scripts/ocr/test_*.py 1> /dev/null 2>&1; then
    mv scripts/ocr/test_*.py src/mindnlp/ocr/tests/
    echo "  ✓ test_*.py → tests/"
fi

# 5. 移动工具文件
echo ""
echo "📦 移动工具文件..."
for file in check_config convert_datasets convert_features_to_numpy ocr_toolkit; do
    if [ -f "scripts/ocr/${file}.py" ]; then
        mv "scripts/ocr/${file}.py" src/mindnlp/ocr/tools/
        echo "  ✓ ${file}.py → tools/"
    fi
done

# 6. 移动 shell 脚本
echo ""
echo "📦 移动 shell 脚本..."
if ls scripts/ocr/*.sh 1> /dev/null 2>&1; then
    mv scripts/ocr/*.sh src/mindnlp/ocr/scripts/
    echo "  ✓ *.sh → scripts/"
fi

if ls scripts/ocr/*.ps1 1> /dev/null 2>&1; then
    mv scripts/ocr/*.ps1 src/mindnlp/ocr/scripts/
    echo "  ✓ *.ps1 → scripts/"
fi

# 7. 移动 README
echo ""
echo "📦 移动文档..."
if [ -f "scripts/ocr/README.md" ]; then
    mv scripts/ocr/README.md src/mindnlp/ocr/docs/scripts_guide.md
    echo "  ✓ README.md → docs/scripts_guide.md"
fi

# 8. 创建 __init__.py 文件
echo ""
echo "📝 创建 __init__.py 文件..."

cat > src/mindnlp/ocr/benchmarks/__init__.py << 'EOF'
"""
Benchmarking tools for OCR model performance evaluation.

This package contains scripts for testing and comparing model performance:
- benchmark_kv_cache.py: KV Cache performance benchmarking
- benchmark_comparison.py: Compare KV Cache enabled vs disabled
- validate_acceptance_criteria.py: Validate performance acceptance criteria
"""

__all__ = [
    'benchmark_kv_cache',
    'benchmark_comparison',
    'validate_acceptance_criteria',
]
EOF
echo "  ✓ benchmarks/__init__.py"

cat > src/mindnlp/ocr/tests/__init__.py << 'EOF'
"""
Test suite for OCR models and features.

This package contains integration and functional tests:
- test_kv_cache.py: KV Cache functionality tests
- test_lora_loading.py: LoRA model loading tests
- test_server_kv_cache.py: Server-side KV Cache tests
"""

__all__ = [
    'test_kv_cache',
    'test_lora_loading',
    'test_server_kv_cache',
]
EOF
echo "  ✓ tests/__init__.py"

cat > src/mindnlp/ocr/tools/__init__.py << 'EOF'
"""
Utility tools for OCR model development and maintenance.

This package contains various development and debugging tools:
- check_config.py: Configuration validation
- convert_datasets.py: Dataset format conversion
- convert_features_to_numpy.py: Feature extraction and conversion
- ocr_toolkit.py: General OCR toolkit utilities
"""

__all__ = [
    'check_config',
    'convert_datasets',
    'convert_features_to_numpy',
    'ocr_toolkit',
]
EOF
echo "  ✓ tools/__init__.py"

# 9. 检查剩余文件
echo ""
echo "🔍 检查 scripts/ocr/ 剩余文件..."
remaining=$(ls -A scripts/ocr 2>/dev/null | wc -l)
if [ $remaining -gt 0 ]; then
    echo "⚠️  scripts/ocr/ 还有 $remaining 个文件/目录:"
    ls -lh scripts/ocr/
else
    echo "✅ scripts/ocr/ 已清空"
fi

# 10. 验证新结构
echo ""
echo "✅ 文件整理完成！"
echo ""
echo "📁 新的目录结构:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

dirs=(
    "src/mindnlp/ocr/benchmarks"
    "src/mindnlp/ocr/tests"
    "src/mindnlp/ocr/tools"
    "src/mindnlp/ocr/scripts"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        count=$(ls -1 "$dir"/*.py 2>/dev/null | wc -l)
        sh_count=$(ls -1 "$dir"/*.sh 2>/dev/null | wc -l)
        ps1_count=$(ls -1 "$dir"/*.ps1 2>/dev/null | wc -l)
        total=$((count + sh_count + ps1_count))
        echo "  ✓ $dir/ ($total 文件)"
        ls "$dir"/ | sed 's/^/      /'
    fi
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📝 后续操作:"
echo "  1. 验证功能: python src/mindnlp/ocr/tests/test_kv_cache.py"
echo "  2. 运行性能测试: python src/mindnlp/ocr/benchmarks/benchmark_comparison.py --help"
echo "  3. 查看目录文档: ls src/mindnlp/ocr/docs/"
