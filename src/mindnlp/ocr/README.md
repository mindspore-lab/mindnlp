# MindNLP OCR Module

基于 Vision-Language Model (VLM) 的 OCR 模块，提供端到端的文字识别、文档理解、表格识别等功能。

## 📁 目录结构

```
src/mindnlp/ocr/
├── api/                    # FastAPI 应用层
│   ├── app.py             # FastAPI 应用工厂
│   └── routes/            # API 路由
│       └── ocr.py         # OCR 相关端点
├── config/                 # 配置管理
│   ├── settings.py        # 应用配置
│   └── prompts.yaml       # Prompt 模板
├── core/                   # 核心业务逻辑
│   ├── engine.py          # VLM OCR 引擎
│   ├── mock_engine.py     # Mock 引擎（测试用）
│   ├── processor/         # 数据处理器
│   │   ├── image.py       # 图像预处理
│   │   ├── prompt.py      # Prompt 构建
│   │   └── batch.py       # 批处理
│   └── validator/         # 输入验证
│       └── input.py       # 请求验证
├── models/                 # 模型封装
│   └── qwen2vl.py         # Qwen2-VL 模型
├── utils/                  # 工具函数
│   └── logger.py          # 日志工具
├── main.py                # 服务入口
└── README.md              # 本文件
```

## 🚀 功能特点

- **轻耦合**: 作为 MindNLP 的子模块，可独立运行
- **标准化**: 完整的 RESTful API，易于集成
- **模块化**: 清晰的分层架构，易于维护和扩展
- **灵活配置**: 支持环境变量和配置文件

## 📋 支持的功能

### OCR 任务类型
- ✅ **通用 OCR** (general) - 识别图像中的所有文本
- ✅ **文档理解** (document) - 解析文档结构和内容
- ✅ **表格识别** (table) - 提取表格数据
- ✅ **公式识别** (formula) - 识别数学公式

### 输出格式
- 📝 **text** - 纯文本格式
- 📊 **json** - JSON 格式（包含文本和坐标）
- 📄 **markdown** - Markdown 格式（保持文档结构）

### 支持的模型
- ✅ **Qwen2-VL-2B-Instruct** - 通用视觉语言模型
- 🔄 InternVL 系列（开发中）
- 📝 LLaVA 系列（计划中）

## 🛠️ 快速开始

### 1. 安装依赖

```bash
# 在 mindnlp 根目录
pip install -r requirements.txt

# OCR 模块额外依赖
pip install -r requirements/ocr-requirements.txt
```

### 2. 配置环境变量（可选）

创建 `src/mindnlp/ocr/.env` 文件：

```bash
# 使用 Mock 引擎进行快速测试（无需下载模型）
OCR_USE_MOCK_ENGINE=True

# API 配置
OCR_API_HOST=0.0.0.0
OCR_API_PORT=8000

# 模型配置（使用真实模型时）
OCR_DEFAULT_MODEL=Qwen/Qwen2-VL-2B-Instruct
```

### 3. 启动服务

```bash
# 方式 1: 从 OCR 目录启动
cd src/mindnlp/ocr
python main.py

# 方式 2: 从 mindnlp 根目录启动
python -m mindnlp.ocr.main
```

服务启动后访问：
- **API 文档**: http://localhost:8000/api/docs
- **健康检查**: http://localhost:8000/api/v1/health

## 📝 API 使用示例

### 1. 健康检查

```bash
curl http://localhost:8000/api/v1/health
```

### 2. 单图 OCR

```bash
curl -X POST http://localhost:8000/api/v1/ocr/predict \
  -F "file=@image.jpg" \
  -F "output_format=text" \
  -F "language=zh" \
  -F "task_type=general"
```

### 3. URL 图像 OCR

```bash
curl -X POST http://localhost:8000/api/v1/ocr/predict-url \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "output_format": "json",
    "language": "auto",
    "task_type": "document"
  }'
```

### 4. 批量 OCR

```bash
curl -X POST http://localhost:8000/api/v1/ocr/predict-batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "output_format=markdown" \
  -F "language=en"
```

## 🐍 Python 客户端示例

```python
import requests
from pathlib import Path

# OCR API 客户端类
class OCRClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict(self, image_path, output_format="text", language="auto", task_type="general"):
        """单图预测"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'output_format': output_format,
                'language': language,
                'task_type': task_type
            }
            response = requests.post(
                f'{self.base_url}/api/v1/ocr/predict',
                files=files,
                data=data
            )
            return response.json()
    
    def predict_url(self, image_url, **kwargs):
        """URL 图像预测"""
        data = {'image_url': image_url, **kwargs}
        response = requests.post(
            f'{self.base_url}/api/v1/ocr/predict-url',
            json=data
        )
        return response.json()
    
    def predict_batch(self, image_paths, **kwargs):
        """批量预测"""
        files = [('files', open(p, 'rb')) for p in image_paths]
        try:
            response = requests.post(
                f'{self.base_url}/api/v1/ocr/predict-batch',
                files=files,
                data=kwargs
            )
            return response.json()
        finally:
            for _, f in files:
                f.close()

# 使用示例
client = OCRClient()

# 1. 通用 OCR
result = client.predict('document.jpg', output_format='text', language='zh')
print(result['text'])

# 2. 表格识别
result = client.predict('table.png', output_format='json', task_type='table')
print(result['structured_output'])

# 3. URL 图像
result = client.predict_url('https://example.com/image.jpg')
print(result)

# 4. 批量处理
results = client.predict_batch(['img1.jpg', 'img2.jpg'], output_format='markdown')
for i, result in enumerate(results['results']):
    print(f"Image {i+1}: {result['text']}")
```

## 🧪 测试

```bash
# 运行所有测试
cd tests/mindnlp/ocr
pytest -v

# 运行特定测试
pytest test_preprocessing.py -v
pytest test_api_complete.py -v

# 查看测试覆盖率
pytest --cov=mindnlp.ocr --cov-report=html
```

## 🔧 开发模式

### Mock Engine

使用 Mock Engine 可以快速测试 API，无需下载大型模型：

```python
# 在 .env 文件中设置
OCR_USE_MOCK_ENGINE=True
```

Mock Engine 会返回模拟的 OCR 结果，适合：
- API 功能测试
- 前端集成开发
- CI/CD 流程

### 真实模型

```python
# 在 .env 文件中设置
OCR_USE_MOCK_ENGINE=False
OCR_DEFAULT_MODEL=Qwen/Qwen2-VL-2B-Instruct
```

首次运行会自动下载模型到 `~/.cache/huggingface/`

## 📊 架构设计

### 分层架构

```
┌─────────────────────────────────────┐
│         API Layer (FastAPI)         │  # RESTful 接口
├─────────────────────────────────────┤
│      Business Logic (Engine)        │  # 业务逻辑
├─────────────────────────────────────┤
│    Processors & Validators          │  # 数据处理
├─────────────────────────────────────┤
│      Model Layer (Qwen2-VL)         │  # 模型封装
└─────────────────────────────────────┘
```

### 核心组件

1. **VLMOCREngine**: 主引擎，协调各个组件
2. **ImageProcessor**: 图像预处理（缩放、padding、归一化）
3. **PromptBuilder**: 构建任务特定的 Prompt
4. **BatchCollator**: 批量数据整理
5. **InputValidator**: 请求参数验证

## 📖 相关文档

- [Issue #2348 - VLM-OCR 模块开发](https://github.com/mindspore-lab/mindnlp/issues/2348)
- [Issue #2349 - API 服务层](https://github.com/mindspore-lab/mindnlp/issues/2349)
- [Issue #2350 - 核心预处理组件](https://github.com/mindspore-lab/mindnlp/issues/2350)

## 📂 项目规范说明

### 目录结构规范

OCR 模块遵循 MindNLP 项目规范：

1. **✅ 模块位置**: `src/mindnlp/ocr/` - 作为 mindnlp 的子模块
2. **✅ 测试位置**: `tests/mindnlp/ocr/` - 测试文件统一放置
3. **✅ 配置文件**: `configs/` - 全局配置（Prometheus、Grafana、Logging）
4. **✅ 依赖管理**: OCR专用依赖在 `requirements/ocr-requirements.txt`
5. **✅ 无独立包**: 不使用单独的 setup.py，统一使用 mindnlp 包管理

### 安装说明

```bash
# 基础依赖（MindNLP核心）
pip install -r requirements/requirements.txt

# OCR模块依赖（使用OCR功能时需要）
pip install -r requirements/ocr-requirements.txt
```

### 不应提交的文件

以下生成文件已通过 `.gitignore` 排除，请勿提交：
- `*.prof` - CPU/Memory profiling 结果
- `*.log` - 日志文件
- `benchmark_*.json` - 性能测试报告
- `*_results.json` - 评估结果文件

### 依赖说明

OCR 模块的核心依赖：
- **API 服务**: FastAPI, Uvicorn, Pydantic
- **图像处理**: OpenCV, Pillow
- **监控日志**: Structlog, OpenTelemetry, Prometheus
- **模型推理**: Transformers, QWen-VL-Utils
- **性能分析**: psutil, tensorboard

完整依赖列表见 `requirements/ocr-requirements.txt`

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

Apache License 2.0

## 🔗 相关链接

- [MindNLP 主仓库](https://github.com/mindspore-lab/mindnlp)
- [Qwen2-VL 模型](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
