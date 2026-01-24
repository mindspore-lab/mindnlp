# 贡献指南

感谢您有兴趣为 MindNLP 做出贡献！本指南将帮助您了解开发流程。

## 开发环境配置

### 1. Fork 并克隆仓库

```bash
# 在 GitHub 上 Fork 仓库，然后克隆您的 Fork
git clone https://github.com/YOUR_USERNAME/mindnlp.git
cd mindnlp

# 添加上游远程仓库
git remote add upstream https://github.com/mindspore-lab/mindnlp.git
```

### 2. 创建开发环境

```bash
# 创建并激活 conda 环境
conda create -n mindnlp python=3.10
conda activate mindnlp

# 安装 MindSpore（根据您的平台选择合适的版本）
# 参见：https://www.mindspore.cn/install

# 以开发模式安装 MindNLP
pip install -e .

# 安装开发依赖
pip install -r requirements/dev_requirements.txt
```

### 3. 保持 Fork 更新

```bash
git fetch upstream
git checkout master
git merge upstream/master
```

## 代码风格指南

### Python 风格

- 遵循 [PEP 8](https://peps.python.org/pep-0008/) 规范
- 使用 4 个空格缩进（不使用制表符）
- 最大行长度：120 个字符
- 使用有意义的变量和函数名

### 文档字符串

使用 Google 风格的文档字符串：

```python
def function_name(param1, param2):
    """函数的简短描述。

    如需要可添加更详细的描述。

    Args:
        param1 (type): param1 的描述。
        param2 (type): param2 的描述。

    Returns:
        type: 返回值的描述。

    Raises:
        ExceptionType: 何时会抛出此异常。
    """
    pass
```

### 导入顺序

按以下顺序组织导入：

1. 标准库导入
2. 第三方库导入
3. MindSpore 导入
4. MindNLP 导入

```python
import os
import sys

import numpy as np

import mindspore
from mindspore import ops

from mindnlp.transformers import AutoModel
```

## 测试

### 运行测试

```bash
# 运行特定测试文件
python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py

# 运行特定测试用例
python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model
```

### 编写测试

- 将测试放在 `tests/` 目录中
- 使用 pytest 编写测试用例
- 确保测试可重复且相互独立

## Pull Request 流程

### 1. 创建功能分支

```bash
git checkout -b feature/your-feature-name
```

### 2. 进行更改

- 编写清晰、文档完善的代码
- 为新功能添加测试
- 确保现有测试通过

### 3. 提交更改

编写清晰、描述性的提交信息：

```bash
git add .
git commit -m "feat: 添加新模型架构支持

- 实现了 NewModel 类
- 添加了分词器支持
- 包含单元测试"
```

遵循约定式提交格式：
- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更改
- `test:` 测试添加/更改
- `refactor:` 代码重构

### 4. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request，包含：

- 描述更改的清晰标题
- 更改内容和原因的描述
- 相关 issue 的引用

### 5. 代码审查

- 及时响应审查反馈
- 在新提交中进行请求的更改
- 保持 PR 专注于单个功能/修复

## 目录结构

```
mindnlp/
├── src/
│   ├── mindnlp/          # MindNLP 主要源代码
│   │   ├── transformers/ # Transformer 模型
│   │   ├── engine/       # 训练引擎
│   │   └── dataset/      # 数据集工具
│   └── mindtorch/        # PyTorch 兼容层
├── tests/                # 测试文件
├── docs/                 # 文档
└── examples/             # 示例脚本
```

## 获取帮助

- 提交 issue 报告 bug 或功能请求
- 加入 MindSpore NLP SIG 参与讨论
- 创建新 issue 前请先查看现有 issue

## 许可证

通过为 MindNLP 做出贡献，您同意您的贡献将根据 Apache 2.0 许可证进行许可。
