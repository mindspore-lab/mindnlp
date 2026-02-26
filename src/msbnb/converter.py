"""
Model Conversion Tools

Provides functionality to automatically replace Linear layers in models with quantized layers.
"""

import numpy as np
try:
    import mindspore as ms
    import mindspore.nn as nn
    from mindspore import Parameter
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False

from .linear import Linear8bit, Linear4bit
from .config import QuantConfig, Int8Config, Int4Config


def convert_to_quantized_model(
    model,
    config=None,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None
):
    """
    Convert Linear layers in model to quantized layers
    
    Args:
        model: Model to convert
        config (QuantConfig, optional): Quantization configuration. Default: Int8Config()
        modules_to_not_convert (list, optional): List of module names to exclude from conversion. Default: None
        current_key_name (str, optional): Current module key name (internal use). Default: None
        quantization_config (dict, optional): Additional quantization configuration. Default: None
    
    Returns:
        model: Converted model
    
    Examples:
        >>> from msbnb import convert_to_quantized_model, Int8Config
        >>> model = YourModel()
        >>> config = Int8Config(symmetric=True, per_channel=True)
        >>> quant_model = convert_to_quantized_model(
        ...     model,
        ...     config=config,
        ...     modules_to_not_convert=["lm_head", "classifier"]
        ... )
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required for model conversion")
    
    if config is None:
        config = Int8Config()
    
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    
    # Recursively replace all Linear layers
    for name, module in model.name_cells().items():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        
        # Check if in exclusion list
        full_name = ".".join(current_key_name)
        if any(key in full_name for key in modules_to_not_convert):
            current_key_name.pop()
            continue
        
        # Check if it's a Linear/Dense layer
        if isinstance(module, (nn.Dense,)):
            # Select quantized layer based on config
            if isinstance(config, Int8Config):
                quant_layer = _convert_to_linear8bit(module, config)
            elif isinstance(config, Int4Config):
                quant_layer = _convert_to_linear4bit(module, config)
            else:
                raise ValueError(f"Unsupported config type: {type(config)}")
            
            # Replace layer
            setattr(model, name, quant_layer)
            print(f"Converted: {full_name} -> {type(quant_layer).__name__}")
        
        # Recursively process submodules
        elif len(list(module.cells())) > 0:
            convert_to_quantized_model(
                module,
                config=config,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config
            )
        
        current_key_name.pop()
    
    return model


def _convert_to_linear8bit(linear_layer, config):
    """Convert Linear layer to Linear8bit"""
    # Get layer parameters
    if hasattr(linear_layer, 'in_channels'):
        in_features = linear_layer.in_channels
        out_features = linear_layer.out_channels
        has_bias = linear_layer.has_bias
    else:
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        has_bias = linear_layer.bias is not None
    
    # Create Linear8bit layer
    quant_layer = Linear8bit(
        in_features=in_features,
        out_features=out_features,
        bias=has_bias,
        has_fp16_weights=config.has_fp16_weights,
        threshold=config.threshold,
        per_channel=config.per_channel,
        symmetric=config.symmetric
    )
    
    # Copy weights
    quant_layer.weight.set_data(linear_layer.weight.data)
    if has_bias:
        quant_layer.bias.set_data(linear_layer.bias.data)
    
    # If not keeping FP16 weights, quantize immediately
    if not config.has_fp16_weights:
        quant_layer.quantize_weights()
    
    return quant_layer


def _convert_to_linear4bit(linear_layer, config):
    """Convert Linear layer to Linear4bit"""
    # Use from_linear method
    quant_layer = Linear4bit.from_linear(
        linear_layer,
        group_size=config.group_size,
        compress_statistics=config.compress_statistics
    )
    
    return quant_layer


def replace_linear_layers(
    model,
    target_class,
    in_features=None,
    out_features=None,
    **kwargs
):
    """
    Recursively replace all Linear layers in model
    
    Args:
        model: Model to process
        target_class: Target quantized layer class (Linear8bit or Linear4bit)
        in_features (int, optional): Only replace layers with specified input dimension. Default: None
        out_features (int, optional): Only replace layers with specified output dimension. Default: None
        **kwargs: Additional parameters to pass to target class
    
    Returns:
        model: Processed model
    
    Examples:
        >>> from msbnb import replace_linear_layers, Linear8bit
        >>> model = YourModel()
        >>> model = replace_linear_layers(
        ...     model,
        ...     Linear8bit,
        ...     has_fp16_weights=False,
        ...     symmetric=True
        ... )
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required for layer replacement")
    
    for name, module in model.name_cells().items():
        # Check if it's a Linear/Dense layer
        if isinstance(module, (nn.Dense,)):
            # Get layer parameters
            if hasattr(module, 'in_channels'):
                layer_in = module.in_channels
                layer_out = module.out_channels
                has_bias = module.has_bias
            else:
                layer_in = module.in_features
                layer_out = module.out_features
                has_bias = module.bias is not None
            
            # Check dimension filter
            if in_features is not None and layer_in != in_features:
                continue
            if out_features is not None and layer_out != out_features:
                continue
            
            # Create quantized layer
            if target_class == Linear8bit:
                quant_layer = Linear8bit(
                    in_features=layer_in,
                    out_features=layer_out,
                    bias=has_bias,
                    **kwargs
                )
                # Copy weights
                quant_layer.weight.set_data(module.weight.data)
                if has_bias:
                    quant_layer.bias.set_data(module.bias.data)
            elif target_class == Linear4bit:
                quant_layer = Linear4bit.from_linear(module, **kwargs)
            else:
                raise ValueError(f"Unsupported target class: {target_class}")
            
            # Replace layer
            setattr(model, name, quant_layer)
            print(f"Replaced: {name} ({layer_in} -> {layer_out})")
        
        # Recursively process submodules
        elif len(list(module.cells())) > 0:
            replace_linear_layers(
                module,
                target_class,
                in_features=in_features,
                out_features=out_features,
                **kwargs
            )
    
    return model


def quantize_model_weights(model, num_bits=8):
    """
    Quantize weights of all quantized layers in model
    
    Args:
        model: Model
        num_bits (int): Quantization bit width (8 or 4). Default: 8
    
    Returns:
        model: Quantized model
    
    Examples:
        >>> model = quantize_model_weights(model, num_bits=8)
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required")
    
    count = 0
    for name, module in model.name_cells().items():
        if isinstance(module, Linear8bit):
            if module.has_fp16_weights:
                module.quantize_weights()
                count += 1
                print(f"Quantized: {name}")
        elif isinstance(module, Linear4bit):
            # Linear4bit is already quantized
            pass
        elif len(list(module.cells())) > 0:
            # Recursively process submodules
            quantize_model_weights(module, num_bits=num_bits)
    
    if count > 0:
        print(f"\nTotal quantized layers: {count}")
    
    return model


def get_model_size(model):
    """
    Get model parameter size
    
    Args:
        model: Model
    
    Returns:
        dict: Model size information
            - total_params: Total number of parameters
            - total_size_mb: Total size (MB)
            - layer_sizes: Size of each layer
    
    Examples:
        >>> size_info = get_model_size(model)
        >>> print(f"Model size: {size_info['total_size_mb']:.2f} MB")
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required")
    
    total_params = 0
    total_size = 0
    layer_sizes = {}
    
    for name, param in model.parameters_and_names():
        param_count = param.data.size
        param_size = param.data.nbytes
        
        total_params += param_count
        total_size += param_size
        
        layer_sizes[name] = {
            'params': param_count,
            'size_mb': param_size / (1024 * 1024),
            'dtype': str(param.dtype)
        }
    
    return {
        'total_params': total_params,
        'total_size_mb': total_size / (1024 * 1024),
        'layer_sizes': layer_sizes
    }


def compare_model_sizes(model_fp, model_quant):
    """
    Compare sizes of FP model and quantized model
    
    Args:
        model_fp: FP model
        model_quant: Quantized model
    
    Returns:
        dict: Comparison results
    
    Examples:
        >>> comparison = compare_model_sizes(fp_model, quant_model)
        >>> print(f"Memory saved: {comparison['memory_saved_percent']:.1f}%")
    """
    size_fp = get_model_size(model_fp)
    size_quant = get_model_size(model_quant)
    
    memory_saved = size_fp['total_size_mb'] - size_quant['total_size_mb']
    memory_saved_percent = (memory_saved / size_fp['total_size_mb']) * 100
    
    return {
        'fp_size_mb': size_fp['total_size_mb'],
        'quant_size_mb': size_quant['total_size_mb'],
        'memory_saved_mb': memory_saved,
        'memory_saved_percent': memory_saved_percent,
        'compression_ratio': size_fp['total_size_mb'] / size_quant['total_size_mb']
    }


def print_quantization_summary(model):
    """
    Print model quantization summary
    
    Args:
        model: Model
    
    Examples:
        >>> print_quantization_summary(model)
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required")
    
    int8_count = 0
    int4_count = 0
    fp_count = 0
    
    for name, module in model.name_cells().items():
        if isinstance(module, Linear8bit):
            int8_count += 1
        elif isinstance(module, Linear4bit):
            int4_count += 1
        elif isinstance(module, (nn.Dense,)):
            fp_count += 1
    
    print("=" * 60)
    print("Model Quantization Summary")
    print("=" * 60)
    print(f"INT8 quantized layers: {int8_count}")
    print(f"INT4 quantized layers: {int4_count}")
    print(f"FP16/FP32 layers: {fp_count}")
    print(f"Total: {int8_count + int4_count + fp_count}")
    
    # Get model size
    size_info = get_model_size(model)
    print(f"\nModel size: {size_info['total_size_mb']:.2f} MB")
    print(f"Total parameters: {size_info['total_params']:,}")
    print("=" * 60)
