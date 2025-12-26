# VLM-OCR API 测试

## 测试文件

- `tests/test_api_complete.py` - Issue #2349 完整验证测试

## 运行测试

### 直接运行（推荐）

```bash
cd mindnlp-ocr
python tests/test_api_complete.py
```

### 使用 pytest（需要安装 pytest）

```bash
cd mindnlp-ocr
pytest tests/test_api_complete.py -v
```

## 测试内容

### 1. 应用生命周期管理
- Lifespan 函数验证
- 引擎依赖注入测试

### 2. RESTful API 端点
- `GET /api/v1/health` - 健康检查
- `GET /api/v1/health/ready` - 就绪检查
- `POST /api/v1/ocr/predict` - 单图 OCR
- `POST /api/v1/ocr/predict_batch` - 批量 OCR
- `POST /api/v1/ocr/predict_url` - URL OCR

### 3. Schema 验证
- OCRRequest/OCRResponse 验证
- 字段完整性检查
- 类型验证

### 4. 错误处理
- 文件类型验证
- 异常处理器测试

### 5. 中间件
- 日志记录
- CORS 配置

### 6. API 文档
- Swagger UI
- OpenAPI Schema

### 7. 配置管理
- Settings 验证
- 环境变量支持

### 8. 代码结构
- 目录结构验证
- 模块导入测试

## 预期结果

```
============================================================
Issue #2349 API 服务层功能验证
============================================================

[✓] 应用创建
[✓] Lifespan 函数
[✓] 引擎依赖注入
[✓] 健康检查端点
[✓] 就绪检查端点
[✓] 单图 OCR 端点
[✓] 批量 OCR 端点
[✓] URL OCR 端点
[✓] 请求 Schema
[✓] 响应 Schema
[✓] 响应字段验证
[✓] 文件类型验证
[✓] 异常处理器
[✓] 日志中间件
[✓] 处理时间记录
[✓] CORS 跨域支持
[✓] Swagger UI 文档
[✓] OpenAPI Schema
[✓] 引擎依赖注入
[✓] 应用配置
[✓] 所有目录结构

总结: 所有核心功能已实现！✓
```

## 依赖项

```bash
pip install fastapi uvicorn pydantic pillow httpx
```

## 注意事项

- 测试使用 Mock 引擎，不需要实际的 VLM 模型
- 测试会自动创建测试图像
- 输出编码设置为 UTF-8
