# 测试快速参考

## 📋 测试文件结构

```
tests/
├── README.md                 # 测试说明文档
├── test_api_complete.py      # Issue #2349 完整验证测试 (289行)
└── __init__.py
```

## 🚀 运行测试

```bash
# 方法1: 直接运行（推荐）
python tests/test_api_complete.py

# 方法2: 使用 pytest
pytest tests/test_api_complete.py -v
```

## ✅ 测试内容（10大类）

1. **应用生命周期管理** - Lifespan 上下文管理器
2. **RESTful API 端点** - 5个 API 端点
3. **请求/响应 Schema** - Pydantic 模型验证
4. **错误处理** - 4种异常处理器
5. **日志记录** - 请求/响应日志中间件
6. **CORS 配置** - 跨域资源共享
7. **API 文档** - Swagger/ReDoc/OpenAPI
8. **依赖注入** - 引擎实例管理
9. **配置管理** - Settings 类
10. **代码结构** - 目录结构验证

## 📊 测试结果示例

```
======================================================================
  Issue #2349 API 服务层功能验证
======================================================================

[✓] 应用创建 - FastAPI 应用创建成功
[✓] Lifespan 函数 - 应用生命周期管理函数已定义
[✓] 引擎依赖注入 - get_engine() 依赖注入函数已实现
[✓] 健康检查端点 GET /api/v1/health - 状态码: 200
[✓] 就绪检查端点 GET /api/v1/health/ready - 状态码: 200
[✓] 单图 OCR 端点 POST /api/v1/ocr/predict - 状态码: 200
[✓] 批量 OCR 端点 POST /api/v1/ocr/predict_batch - 状态码: 200
...
总结: 所有核心功能已实现！✓
```

## 🔧 依赖项

```bash
pip install fastapi uvicorn pydantic pillow httpx
```

## 📝 注意事项

- ✅ 测试使用 Mock 引擎（不需要实际模型）
- ✅ 自动创建测试图像
- ✅ 输出编码自动设置为 UTF-8
- ✅ 支持直接运行和 pytest 运行
- ✅ 完整覆盖 Issue #2349 所有要求
