# MindNLP OCR模块设计文档

## 1. 设计概述

### 1.1 核心设计原则

**1. 模型层依赖Transformers**
- 直接调用HuggingFace Transformers库的VLM模型
- 支持Qwen2-VL、InternVL、LLaVA等主流模型
- 模型加载和推理使用transformers标准接口

**2. 业务层自研实现**
- 图像预处理、Prompt构建、结果后处理完全自己实现
- API服务层使用FastAPI从零搭建
- 不依赖mindnlp现有的任何模块

**3. 先完成流程,后优化性能**
- Phase 1: 实现完整的端到端流程(本文档重点)
- Phase 2: 引入性能优化(KV Cache、Flash Attention等,后续设计)

## 2. 模块架构设计

### 2.1 总体架构

```
mindnlp-ocr/
├── api/                    # API服务层(自研)
│   ├── app.py             # FastAPI应用入口
│   ├── routes/            # 路由定义
│   │   ├── __init__.py
│   │   ├── ocr.py        # OCR预测接口
│   │   └── health.py     # 健康检查接口
│   ├── schemas/           # 请求/响应模型
│   │   ├── __init__.py
│   │   ├── request.py    # OCRRequest
│   │   └── response.py   # OCRResponse
│   └── middleware/        # 中间件
│       ├── __init__.py
│       ├── error.py      # 错误处理
│       └── logging.py    # 日志记录
│
├── core/                  # 核心业务层(自研)
│   ├── __init__.py
│   ├── engine.py         # VLMOCREngine主引擎
│   ├── processor/        # 预处理器
│   │   ├── __init__.py
│   │   ├── image.py     # ImageProcessor
│   │   ├── prompt.py    # PromptBuilder
│   │   └── batch.py     # BatchCollator
│   ├── parser/           # 后处理器
│   │   ├── __init__.py
│   │   ├── decoder.py   # TokenDecoder
│   │   ├── result.py    # ResultParser
│   │   └── formatter.py # OutputFormatter
│   └── validator/        # 验证器
│       ├── __init__.py
│       └── input.py     # 输入验证
│
├── models/               # 模型层(调用transformers)
│   ├── __init__.py
│   ├── base.py          # VLMModelBase抽象类
│   ├── qwen2vl.py       # Qwen2VL模型封装
│   ├── internvl.py      # InternVL模型封装
│   └── loader.py        # 模型加载器
│
├── utils/               # 工具库(自研)
│   ├── __init__.py
│   ├── image_utils.py   # 图像工具
│   ├── text_utils.py    # 文本工具
│   └── logger.py        # 日志工具
│
├── config/              # 配置管理(自研)
│   ├── __init__.py
│   ├── settings.py      # 配置类
│   └── prompts.yaml     # Prompt模板
│
├── tests/               # 测试
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_processor.py
│   └── test_models.py
│
├── requirements.txt     # 依赖列表
├── setup.py            # 安装脚本
├── README.md           # 使用文档
└── main.py             # 启动入口
```

### 2.2 层级职责划分

| 层级 | 职责 | 实现方式 | 依赖 |
|-----|------|---------|------|
| **API层** | 接收HTTP请求,返回结果 | FastAPI自研 | FastAPI, Pydantic |
| **核心层** | 预处理、后处理、流程编排 | 完全自研 | PIL, NumPy, OpenCV |
| **模型层** | VLM模型推理 | **调用transformers** | transformers, torch |
| **工具层** | 通用工具函数 | 完全自研 | 标准库 |

### 2.3 数据流转

```
HTTP请求 (POST /api/v1/ocr/predict)
    ↓
[API层] 请求验证 & 参数解析 (FastAPI)
    ↓
[核心层] ImageProcessor.process()
    ├─ 图像加载 (PIL)
    ├─ 尺寸归一化 (自研)
    ├─ 数值标准化 (NumPy)
    └─ Tensor转换 → [B,3,H,W]
    ↓
[核心层] PromptBuilder.build()
    ├─ 任务类型识别 (自研)
    ├─ Prompt模板填充 (自研)
    └─ 返回Prompt字符串
    ↓
[模型层] VLMModel.generate()
    ├─ transformers.AutoModel.from_pretrained()
    ├─ model.generate(pixel_values, prompt)  ← **使用transformers**
    └─ 返回Token IDs
    ↓
[模型层] VLMModel.decode()
    └─ tokenizer.decode(token_ids)  ← **使用transformers**
    ↓
[核心层] ResultParser.parse()
    ├─ 格式识别 (JSON/Text/Markdown)
    ├─ 结构化提取 (自研正则/JSON解析)
    └─ 返回OCRResult对象
    ↓
[核心层] OutputFormatter.format()
    ├─ 坐标映射 (自研)
    ├─ 置信度过滤 (自研)
    └─ 结果排序 (自研)
    ↓
[API层] 响应封装 & 返回 (Pydantic)
    ↓
HTTP响应 (JSON)
```