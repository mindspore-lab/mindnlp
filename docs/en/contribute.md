# Contributing to MindNLP

Thank you for your interest in contributing to MindNLP! This guide will help you get started with the development process.

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mindnlp.git
cd mindnlp

# Add upstream remote
git remote add upstream https://github.com/mindspore-lab/mindnlp.git
```

### 2. Create a Development Environment

```bash
# Create and activate a conda environment
conda create -n mindnlp python=3.10
conda activate mindnlp

# Install MindSpore (choose the appropriate version for your platform)
# See: https://www.mindspore.cn/install

# Install MindNLP in development mode
pip install -e .

# Install development dependencies
pip install -r requirements/dev_requirements.txt
```

### 3. Keep Your Fork Updated

```bash
git fetch upstream
git checkout master
git merge upstream/master
```

## Code Style Guidelines

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use meaningful variable and function names

### Docstrings

Use Google-style docstrings for functions and classes:

```python
def function_name(param1, param2):
    """Short description of function.

    Longer description if needed.

    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.

    Returns:
        type: Description of return value.

    Raises:
        ExceptionType: When this exception is raised.
    """
    pass
```

### Import Order

Organize imports in the following order:

1. Standard library imports
2. Third-party imports
3. MindSpore imports
4. MindNLP imports

```python
import os
import sys

import numpy as np

import mindspore
from mindspore import ops

from mindnlp.transformers import AutoModel
```

## Testing

### Running Tests

```bash
# Run specific test file
python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py

# Run specific test case
python tests/run_test.py -vs tests/transformers/tests/models/bert/test_modeling_bert.py::BertModelTest::test_model
```

### Writing Tests

- Place tests in the `tests/` directory
- Use pytest for writing test cases
- Ensure tests are reproducible and independent

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clean, well-documented code
- Add tests for new functionality
- Ensure existing tests pass

### 3. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: Add support for new model architecture

- Implemented NewModel class
- Added tokenizer support
- Included unit tests"
```

Follow conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions/changes
- `refactor:` for code refactoring

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- Clear title describing the change
- Description of what was changed and why
- Reference to any related issues

### 5. Code Review

- Respond to review feedback promptly
- Make requested changes in new commits
- Keep the PR focused on a single feature/fix

## Directory Structure

```
mindnlp/
├── src/
│   ├── mindnlp/          # Main MindNLP source code
│   │   ├── transformers/ # Transformer models
│   │   ├── engine/       # Training engine
│   │   └── dataset/      # Dataset utilities
│   └── mindtorch/        # PyTorch compatibility layer
├── tests/                # Test files
├── docs/                 # Documentation
└── examples/             # Example scripts
```

## Getting Help

- Open an issue for bugs or feature requests
- Join the MindSpore NLP SIG for discussions
- Check existing issues before creating new ones

## License

By contributing to MindNLP, you agree that your contributions will be licensed under the Apache 2.0 License.
