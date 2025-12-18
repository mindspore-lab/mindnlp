# MindNLP Patch Architecture

## 目录结构

```
mindnlp/
├── patch/                    # 补丁系统（新）
│   ├── __init__.py
│   ├── registry.py           # 版本化补丁注册表
│   ├── transformers/         # transformers 补丁
│   │   ├── __init__.py
│   │   ├── common.py         # 通用补丁
│   │   ├── v4_55.py          # 4.55.x 版本补丁
│   │   └── v4_56.py          # 4.56.x+ 版本补丁
│   ├── diffusers/            # diffusers 补丁
│   │   ├── __init__.py
│   │   └── common.py
│   └── utils/                # 补丁工具函数
│       ├── masking_utils.py
│       ├── modeling_utils.py
│       ├── cache_utils.py
│       ├── tokenization_utils.py
│       └── trainer.py
│
├── transformers/             # 已废弃（保留用于向后兼容）
│   └── __init__.py          # 显示警告，重定向到 patched transformers
│
├── diffusers/               # 已废弃（保留用于向后兼容）
│   └── __init__.py          # 显示警告，重定向到 patched diffusers
│
└── models/                  # MindSpore 原生实现（新，参考 mlx-lm/mlx-vlm）
    ├── __init__.py
    └── README.md
```

## 使用方式

### 推荐方式（新）

```python
# 1. 导入 mindnlp 自动打补丁
import mindnlp

# 2. 直接使用 transformers/diffusers（已打补丁）
from transformers import AutoModel, AutoTokenizer
from diffusers import DiffusionPipeline

model = AutoModel.from_pretrained("bert-base-uncased")
```

### 原生实现（未来）

```python
# 使用 MindSpore 原生实现（类似 mlx-lm）
from mindnlp.models import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
```

### 已废弃的方式（仍支持但会警告）

```python
# 不推荐：会显示 DeprecationWarning
from mindnlp.transformers import AutoModel
from mindnlp.diffusers import DiffusionPipeline
```

## 补丁系统特性

### 1. 版本化管理

使用 `packaging.specifiers` 进行版本匹配：

```python
@register_transformers_patch(">=4.56.0,<4.57.0", priority=10)
def patch_pre_trained_model():
    # 针对 4.56.x 的补丁
    pass
```

### 2. 优先级和依赖

```python
@register_transformers_patch(">=4.56.0", priority=20,
                            depends_on=["patch_common_features"])
def patch_advanced_feature():
    # 高级补丁，依赖于通用补丁
    pass
```

### 3. 错误隔离

单个补丁失败不影响其他补丁执行。

## 迁移指南

### 从旧方式迁移

**旧代码：**
```python
from mindnlp.transformers import AutoModel
```

**新代码：**
```python
import mindnlp
from transformers import AutoModel
```

### 添加新版本补丁

1. 在 `patch/transformers/` 创建新文件，如 `v4_57.py`
2. 使用装饰器注册补丁：

```python
from ..registry import register_transformers_patch

@register_transformers_patch(">=4.57.0,<4.58.0", priority=10)
def patch_v4_57():
    # 补丁逻辑
    pass
```

3. 在 `patch/transformers/__init__.py` 中导入新模块

## 命名说明

参考 MLX 的命名方式：
- **mlx-lm**: Language Models for MLX
- **mlx-vlm**: Vision-Language Models for MLX
- **mindnlp.models**: Native MindSpore model implementations

## 未来计划

1. 逐步实现 MindSpore 原生模型
2. 提供与 HuggingFace 兼容的 API
3. 优化性能，充分利用 MindSpore 特性
