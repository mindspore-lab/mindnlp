# Changelog

All notable changes to MindNLP are documented here.

## Version 0.6.x (Current)

**MindSpore**: >=2.7.1 | **Python**: 3.10-3.11

### Highlights

- Full HuggingFace Transformers compatibility via patching mechanism
- Full HuggingFace Diffusers compatibility
- Support for latest model architectures (Qwen3, Llama3, etc.)
- Enhanced mindtorch layer for PyTorch API compatibility
- Improved device management and heterogeneous computing support

### New Features

- Automatic patching of `transformers` and `diffusers` libraries
- Support for `ms_dtype` parameter in model loading
- Enhanced `device_map` support for multi-device inference
- Improved tensor serialization and checkpoint handling

## Version 0.5.x

**MindSpore**: 2.5.0-2.7.0 | **Python**: 3.10-3.11

### Highlights

- Major API refactoring for better HuggingFace compatibility
- Introduction of mindtorch compatibility layer
- Support for new model families (Gemma, Phi-3, etc.)

### New Features

- `mindnlp.core` module providing PyTorch-compatible APIs
- Enhanced AutoModel classes for various tasks
- Improved tokenizer support
- PEFT/LoRA integration for parameter-efficient fine-tuning

## Version 0.4.x

**MindSpore**: 2.2.x-2.5.0 | **Python**: 3.9-3.11

### Highlights

- Expanded model support
- Improved training stability
- Enhanced Trainer API

### New Features

- Support for Qwen2, Mistral, Mixtral models
- Enhanced gradient checkpointing
- Improved distributed training support
- Better memory management for large models

## Version 0.3.x

**MindSpore**: 2.1.0-2.3.1 | **Python**: 3.8-3.9

### Highlights

- Stable release with comprehensive model coverage
- Improved documentation and examples

### New Features

- Support for Llama, Llama2 models
- ChatGLM series support (ChatGLM, ChatGLM2, ChatGLM3)
- Enhanced dataset loading utilities
- Improved model serialization

## Version 0.2.x

**MindSpore**: >=2.1.0 | **Python**: 3.8-3.9

### Highlights

- Major architecture improvements
- Better alignment with HuggingFace APIs

### New Features

- Refactored model architecture
- Improved tokenizer implementations
- Enhanced training engine
- Better error messages and debugging

## Version 0.1.x

**MindSpore**: 1.8.1-2.0.0 | **Python**: 3.7.5-3.9

### Highlights

- Initial release of MindNLP
- Core transformer model support

### New Features

- Basic transformer models (BERT, GPT-2, T5, etc.)
- Tokenizer support
- Dataset loading utilities
- Basic training loop implementation

---

For detailed release notes, see [GitHub Releases](https://github.com/mindspore-lab/mindnlp/releases).
