# MindBNB

MindBNB is MindNLP's implementation of BitsAndBytes quantization, enabling 8-bit and 4-bit model quantization for memory-efficient inference.

## Overview

MindBNB provides:

- **8-bit quantization**: Int8 matrix multiplication for reduced memory usage
- **4-bit quantization**: NF4 and FP4 quantization for even smaller models
- **Integration with transformers**: Seamless use with HuggingFace models

## Installation

MindBNB requires building the C++ extensions:

```bash
bash /path/to/mindnlp/src/mindnlp/quant/mindbnb/scripts/build.sh
```

## Usage

### 8-bit Quantization

```python
import mindnlp
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,
    device_map="auto"
)
```

### 4-bit Quantization

```python
import mindnlp
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Components

- **nn**: Neural network layers with quantization support
- **functional**: Functional quantization operations
- **autograd**: Autograd functions for quantized operations

## Notes

- 4-bit quantization significantly reduces memory usage (4x compared to FP16)
- Some accuracy trade-off is expected with quantization
- GPU support is recommended for optimal performance
