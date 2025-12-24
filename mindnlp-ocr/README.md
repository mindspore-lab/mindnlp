# MindNLP VLM-OCR 模块

基于Vision-Language Model (VLM)的OCR模块，支持端到端的文字识别、文档理解、表格识别等多种OCR任务。

## 🚀 功能特点

- **轻耦合**: 与mindnlp其他模块解耦，独立运行
- **标准化**: 完整的RESTful API，易于集成
- **模块化**: 清晰的分层架构，易于维护
- **可扩展**: 支持多种VLM模型(Qwen2-VL、InternVL等)

## 📋 支持的功能

### OCR任务类型
- ✅ 通用OCR - 识别图像中的所有文本
- ✅ 文档理解 - 解析文档结构和内容
- ✅ 表格识别 - 提取表格数据
- ✅ 公式识别 - 识别数学公式

### 输出格式
- 📝 纯文本格式 (text)
- 📊 JSON格式 (包含文本和坐标)
- 📄 Markdown格式 (保持文档结构)

### 支持的模型
- Qwen2-VL系列
- InternVL系列 (开发中)
- LLaVA系列 (计划中)

## 🛠️ 安装

### 1. 从源码安装

```bash
cd mindnlp-ocr
pip install -e .
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 🎯 快速开始

### 1. 启动服务

```bash
# 使用默认配置启动
python main.py

# 或使用环境变量配置
OCR_API_PORT=8080 OCR_DEFAULT_MODEL=Qwen/Qwen2-VL-2B-Instruct python main.py
```

### 2. API调用示例

#### 健康检查
```bash
curl http://localhost:8000/api/v1/health
```

#### 单张图像OCR
```bash
curl -X POST http://localhost:8000/api/v1/ocr/predict \
  -F "file=@image.jpg" \
  -F "output_format=text" \
  -F "language=zh" \
  -F "task_type=general"
```

#### 从URL预测
```bash
curl -X POST http://localhost:8000/api/v1/ocr/predict_url \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "output_format": "json",
    "language": "auto",
    "task_type": "document"
  }'
```

### 3. Python客户端示例

```python
import requests

# 上传图像进行OCR
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'output_format': 'text',
        'language': 'zh',
        'task_type': 'general'
    }
    response = requests.post(
        'http://localhost:8000/api/v1/ocr/predict',
        files=files,
        data=data
    )
    result = response.json()
    print(result['text'])
```

## 📁 项目结构

```
mindnlp-ocr/
├── api/                    # API服务层
│   ├── routes/            # 路由定义
│   ├── schemas/           # 请求/响应模型
│   └── middleware/        # 中间件
├── core/                  # 核心业务层
│   ├── processor/        # 预处理器
│   ├── parser/           # 后处理器
│   └── validator/        # 验证器
├── models/               # 模型层
│   ├── base.py          # 模型基类
│   ├── qwen2vl.py       # Qwen2-VL封装
│   └── loader.py        # 模型加载器
├── utils/               # 工具库
├── config/              # 配置管理
├── tests/               # 测试
└── main.py             # 启动入口
```

## ⚙️ 配置

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OCR_API_HOST` | API服务地址 | `0.0.0.0` |
| `OCR_API_PORT` | API服务端口 | `8000` |
| `OCR_DEFAULT_MODEL` | 默认模型 | `Qwen/Qwen2-VL-2B-Instruct` |
| `OCR_DEVICE` | 运行设备 | `cuda` |
| `OCR_LOG_LEVEL` | 日志级别 | `INFO` |

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_api.py

# 跳过慢速测试
pytest -m "not slow" tests/
```

## 📊 API文档

启动服务后，访问以下地址查看交互式API文档：

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 🔧 开发

### 添加新模型

1. 在 `models/` 目录创建新的模型文件
2. 继承 `VLMModelBase` 基类
3. 实现 `load_model()`, `load_tokenizer()`, `generate()` 方法
4. 在 `models/loader.py` 中注册模型

### 自定义Prompt

编辑 `config/prompts.yaml` 文件，添加或修改Prompt模板。

## 📝 相关Issue

- [#2348](https://github.com/mindspore-lab/mindnlp/issues/2348) - VLM-OCR模块主Issue
- [#2349](https://github.com/mindspore-lab/mindnlp/issues/2349) - API服务层
- [#2350](https://github.com/mindspore-lab/mindnlp/issues/2350) - 预处理组件
- [#2351](https://github.com/mindspore-lab/mindnlp/issues/2351) - 模型层封装
- [#2352](https://github.com/mindspore-lab/mindnlp/issues/2352) - 后处理组件

## 📄 许可证

Apache License 2.0

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

- GitHub Issues: https://github.com/mindspore-lab/mindnlp/issues
- 项目主页: https://github.com/mindspore-lab/mindnlp
