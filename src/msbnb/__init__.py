"""
MindSpore BitsAndBytes (msbnb)

Provides bitsandbytes-style quantization interface based on MindSpore native quantization operators.

Main Features:
- INT8/INT4 weight quantization
- QLoRA training support
- Automatic model conversion
- Memory optimization

Examples:
    >>> from msbnb import Linear8bit, Linear4bit
    >>> # Create INT8 quantized layer
    >>> layer = Linear8bit(768, 3072)
    >>> # Create INT4 quantized layer
    >>> layer = Linear4bit(768, 3072, group_size=128)
    >>> 
    >>> # Model conversion
    >>> from msbnb import convert_to_quantized_model, Int8Config
    >>> config = Int8Config()
    >>> quant_model = convert_to_quantized_model(model, config)
    >>> 
    >>> # QLoRA training
    >>> from msbnb import Linear4bitWithLoRA
    >>> qlora_layer = Linear4bitWithLoRA(768, 3072, r=8, lora_alpha=16)
"""

from .linear import Linear8bit, Linear4bit, LinearQuant
from .config import QuantConfig, Int8Config, Int4Config
from .utils import (
    quantize_weight_int4_pergroup,
    pack_int4_to_qint4x2,
    unpack_qint4x2_to_int8,
    compute_scale_offset
)
from .functional import (
    quantize_8bit,
    dequantize_8bit,
    quantize_4bit,
    dequantize_4bit,
    quantize_tensor,
    dequantize_tensor,
    estimate_quantization_error,
    get_quantization_info
)
from .converter import (
    convert_to_quantized_model,
    replace_linear_layers,
    quantize_model_weights,
    get_model_size,
    compare_model_sizes,
    print_quantization_summary
)
from .lora import (
    LoRALinear,
    Linear4bitWithLoRA,
    Linear8bitWithLoRA,
    freeze_model_except_lora,
    print_lora_info
)

__all__ = [
    # Quantized layers
    'Linear8bit',
    'Linear4bit',
    'LinearQuant',
    # Configurations
    'QuantConfig',
    'Int8Config',
    'Int4Config',
    # Utility functions
    'quantize_weight_int4_pergroup',
    'pack_int4_to_qint4x2',
    'unpack_qint4x2_to_int8',
    'compute_scale_offset',
    # Functional interface
    'quantize_8bit',
    'dequantize_8bit',
    'quantize_4bit',
    'dequantize_4bit',
    'quantize_tensor',
    'dequantize_tensor',
    'estimate_quantization_error',
    'get_quantization_info',
    # Model conversion
    'convert_to_quantized_model',
    'replace_linear_layers',
    'quantize_model_weights',
    'get_model_size',
    'compare_model_sizes',
    'print_quantization_summary',
    # LoRA / QLoRA
    'LoRALinear',
    'Linear4bitWithLoRA',
    'Linear8bitWithLoRA',
    'freeze_model_except_lora',
    'print_lora_info',
]

__version__ = '0.3.0'
