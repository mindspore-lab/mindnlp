"""
OCR API 独立启动脚本
完全独立于 mindnlp 其他模块，避免触发 mindspore 依赖
"""

import os
import sys
import subprocess

# 将 src 目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def check_and_install_dependencies():
    """检测并自动安装缺失的依赖"""
    required_packages = {
        'torch': 'torch>=2.4.0',
        'torchvision': 'torchvision>=0.19.0',
        'transformers': 'transformers>=4.37.0',
        'fastapi': 'fastapi>=0.109.0',
        'uvicorn': 'uvicorn[standard]>=0.27.0',
        'PIL': 'pillow>=10.0.0',
        'pydantic_settings': 'pydantic-settings>=2.0.0',
        'requests': 'requests>=2.31.0',
        'yaml': 'pyyaml>=6.0',
    }
    
    missing_packages = []
    
    print("正在检查依赖...")
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (缺失)")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n发现 {len(missing_packages)} 个缺失的依赖包")
        print("正在自动安装...")
        
        try:
            # 使用当前 Python 解释器安装依赖
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_packages
            subprocess.check_call(cmd)
            print("✓ 依赖安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ 依赖安装失败: {e}")
            print("\n请手动安装依赖:")
            print(f"  pip install {' '.join(missing_packages)}")
            return False
    else:
        print("✓ 所有依赖已满足")
        return True


# 检查并安装依赖
if not check_and_install_dependencies():
    print("\n无法继续启动服务，请先安装依赖")
    sys.exit(1)

print("")  # 空行分隔

# 设置环境变量
os.environ['NO_PROXY'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 直接导入 OCR 子模块，避免 mindnlp.__init__.py
# 使用 sys.modules 技巧来避免 mindnlp.__init__.py 的执行
import types

# 创建一个空的 mindnlp 模块
mindnlp = types.ModuleType('mindnlp')
mindnlp.__path__ = [os.path.join(src_dir, 'mindnlp')]
sys.modules['mindnlp'] = mindnlp

# 创建空的 mindnlp.ocr 模块
mindnlp_ocr = types.ModuleType('mindnlp.ocr')
mindnlp_ocr.__path__ = [current_dir]
sys.modules['mindnlp.ocr'] = mindnlp_ocr

# 现在可以安全导入 OCR 子模块了
from mindnlp.ocr.api.app import create_app
import uvicorn


def main():
    """启动 API 服务"""
    # 加载配置
    from mindnlp.ocr.config.settings import get_settings
    settings = get_settings()
    
    # 创建应用
    app = create_app()
    
    # 启动服务器
    print(f"\n{'='*60}")
    print(f"启动 OCR API 服务...")
    print(f"  - Host: {settings.api_host}")
    print(f"  - Port: {settings.api_port}")
    print(f"  - 设备: {settings.device}")
    print(f"  - 模型: {settings.default_model}")
    print(f"  - Mock模式: {settings.use_mock_engine}")
    print(f"\n📚 API 文档地址 (推荐使用):")
    print(f"  - Swagger UI:  http://localhost:{settings.api_port}/api/docs")
    print(f"  - ReDoc:       http://localhost:{settings.api_port}/api/redoc")
    print(f"\n🔍 API 端点:")
    print(f"  - 健康检查 (GET):    http://localhost:{settings.api_port}/api/v1/health")
    print(f"  - OCR预测 (POST):    http://localhost:{settings.api_port}/api/v1/ocr/predict")
    print(f"  - 批量OCR (POST):    http://localhost:{settings.api_port}/api/v1/ocr/predict_batch")
    print(f"  - URL OCR (POST):    http://localhost:{settings.api_port}/api/v1/ocr/predict_url")
    print(f"\n💡 提示: POST 端点请使用 API 文档页面进行交互式测试")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
