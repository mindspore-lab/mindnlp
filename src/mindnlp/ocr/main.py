"""
OCR API 独立启动脚本
完全独立于 mindnlp 其他模块，避免触发 mindspore 依赖
"""

import os
import sys

# 将 src 目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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
