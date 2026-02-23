"""
LoRA (Low-Rank Adaptation) Implementation

Provides LoRA adapters and QLoRA training support.
"""

import numpy as np
try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Tensor, Parameter
    from mindspore.common import dtype as mstype
    from mindspore.common.initializer import initializer, Normal, Zero
    MINDSPORE_AVAILABLE = True
except ImportError:
    MINDSPORE_AVAILABLE = False
    print("Warning: MindSpore not available, LoRA layers will not work")

from .linear import Linear4bit, LinearQuant


class LoRALinear(nn.Cell if MINDSPORE_AVAILABLE else object):
    """
    LoRA adapter layer
    
    Implements Low-Rank Adaptation by adapting pretrained models through two low-rank matrices A and B.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        r (int): LoRA rank. Default: 8
        lora_alpha (int): LoRA scaling factor. Default: 16
        lora_dropout (float): LoRA dropout probability. Default: 0.0
        merge_weights (bool): Whether to merge weights. Default: False
    
    Inputs:
        - **x** (Tensor) - Input tensor with shape: [batch, ..., in_features]
    
    Outputs:
        - **output** (Tensor) - Output tensor with shape: [batch, ..., out_features]
    
    Examples:
        >>> lora = LoRALinear(768, 3072, r=8, lora_alpha=16)
        >>> x = Tensor(np.random.randn(32, 768), dtype=ms.float32)
        >>> out = lora(x)
    
    Note:
        LoRA output: out = x @ (A @ B) * scaling
        where scaling = lora_alpha / r
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False
    ):
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore is required for LoRALinear")
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        
        # LoRA parameters
        # A: [in_features, r] - initialized with normal distribution
        # B: [r, out_features] - initialized with zeros
        self.lora_A = Parameter(
            initializer(Normal(sigma=0.01), [in_features, r], mstype.float32),
            name='lora_A',
            requires_grad=True
        )
        self.lora_B = Parameter(
            initializer(Zero(), [r, out_features], mstype.float32),
            name='lora_B',
            requires_grad=True
        )
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Dropout
        if lora_dropout > 0.0:
            self.dropout = nn.Dropout(keep_prob=1.0 - lora_dropout)
        else:
            self.dropout = None
        
        # Operators
        self.matmul = ops.MatMul()
        self.cast = ops.Cast()
    
    def construct(self, x):
        """Forward pass"""
        # LoRA path: x @ A @ B * scaling
        # x: [..., in_features]
        # A: [in_features, r]
        # B: [r, out_features]
        
        # Ensure consistent types
        x_dtype = x.dtype
        lora_A = self.cast(self.lora_A, x_dtype)
        lora_B = self.cast(self.lora_B, x_dtype)
        
        # Save original shape
        original_shape = x.shape
        
        # If input is multi-dimensional, reshape to 2D
        if len(original_shape) > 2:
            x_2d = x.reshape(-1, original_shape[-1])
        else:
            x_2d = x
        
        # Apply dropout
        if self.dropout is not None and self.training:
            x_lora = self.dropout(x_2d)
        else:
            x_lora = x_2d
        
        # x @ A -> [..., r]
        out_A = self.matmul(x_lora, lora_A)
        
        # (x @ A) @ B -> [..., out_features]
        out_B = self.matmul(out_A, lora_B)
        
        # Apply scaling
        out = out_B * self.scaling
        
        # Restore original shape
        if len(original_shape) > 2:
            out = out.reshape(*original_shape[:-1], self.out_features)
        
        return out
    
    def get_merged_weight(self):
        """
        Get merged weight delta
        
        Returns:
            np.ndarray: Weight delta [out_features, in_features]
        """
        # delta_W = B @ A^T * scaling
        # A: [in_features, r]
        # B: [r, out_features]
        # delta_W: [out_features, in_features]
        
        A = self.lora_A.asnumpy()  # [in_features, r]
        B = self.lora_B.asnumpy()  # [r, out_features]
        
        # B @ A^T
        delta_W = np.matmul(B.T, A.T) * self.scaling  # [out_features, in_features]
        
        return delta_W


class Linear4bitWithLoRA(Linear4bit):
    """
    4-bit quantization + LoRA layer (QLoRA)
    
    Combines INT4 quantization with LoRA adapter for efficient large model fine-tuning.
    Quantized weights are frozen, only LoRA parameters are trained.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        bias (bool): Whether to use bias. Default: True
        compute_dtype (mstype): Computation data type. Default: mstype.float32
        compress_statistics (bool): Whether to compress statistics. Default: True
        group_size (int): Group size for per-group quantization. Default: 128
        r (int): LoRA rank. Default: 8
        lora_alpha (int): LoRA scaling factor. Default: 16
        lora_dropout (float): LoRA dropout probability. Default: 0.0
    
    Inputs:
        - **x** (Tensor) - Input tensor with shape: [batch, ..., in_features]
    
    Outputs:
        - **output** (Tensor) - Output tensor with shape: [batch, ..., out_features]
    
    Examples:
        >>> # Create QLoRA layer
        >>> layer = Linear4bitWithLoRA(768, 3072, r=8, lora_alpha=16)
        >>> x = Tensor(np.random.randn(32, 768), dtype=ms.float32)
        >>> out = layer(x)
        >>> 
        >>> # Convert from existing layer
        >>> fp16_layer = nn.Dense(768, 3072)
        >>> qlora_layer = Linear4bitWithLoRA.from_linear(
        ...     fp16_layer, r=8, lora_alpha=16
        ... )
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: 'mstype' = None,
        compress_statistics: bool = True,
        group_size: int = 128,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0
    ):
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore is required for Linear4bitWithLoRA")
        
        # Initialize Linear4bit
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            compute_dtype=compute_dtype or mstype.float32,
            compress_statistics=compress_statistics,
            group_size=group_size
        )
        
        # Freeze quantized weights
        self.weight.requires_grad = False
        self.scale.requires_grad = False
        self.offset.requires_grad = False
        if self.scale_scale is not None:
            self.scale_scale.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # Add LoRA adapter
        self.lora = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Save LoRA configuration
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Add cast operator
        self.cast = ops.Cast()
    
    def construct(self, x):
        """Forward pass"""
        # Ensure consistent input type
        x_dtype = x.dtype
        
        # Main path: INT4 quantized computation
        out_main = super().construct(x)
        
        # LoRA path: low-rank adaptation
        out_lora = self.lora(x)
        
        # Ensure consistent types before merging
        out_main = self.cast(out_main, x_dtype)
        out_lora = self.cast(out_lora, x_dtype)
        
        # Merge outputs
        out = out_main + out_lora
        
        return out

    
    @classmethod
    def from_linear(
        cls,
        linear_layer,
        group_size=128,
        compress_statistics=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0
    ):
        """
        Convert standard Linear layer to Linear4bitWithLoRA
        
        Args:
            linear_layer: nn.Dense or nn.Linear layer
            group_size (int): Quantization group size. Default: 128
            compress_statistics (bool): Whether to use double quantization. Default: True
            r (int): LoRA rank. Default: 8
            lora_alpha (int): LoRA scaling factor. Default: 16
            lora_dropout (float): LoRA dropout probability. Default: 0.0
        
        Returns:
            Linear4bitWithLoRA: QLoRA layer
        
        Examples:
            >>> fp16_layer = nn.Dense(768, 3072)
            >>> qlora_layer = Linear4bitWithLoRA.from_linear(
            ...     fp16_layer, r=8, lora_alpha=16
            ... )
        """
        # Get layer parameters
        if hasattr(linear_layer, 'in_channels'):
            in_features = linear_layer.in_channels
            out_features = linear_layer.out_channels
            has_bias = linear_layer.has_bias
        else:
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            has_bias = linear_layer.bias is not None
        
        # Create Linear4bitWithLoRA instance
        qlora_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            group_size=group_size,
            compress_statistics=compress_statistics,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Quantize weights (using parent class method)
        from .utils import quantize_weight_int4_pergroup
        
        weight_fp16 = linear_layer.weight.data.asnumpy()
        weight_int4, scale, offset = quantize_weight_int4_pergroup(
            weight_fp16, group_size, compress_statistics
        )
        
        # Set quantization parameters
        qlora_layer.weight.set_data(Tensor(weight_int4, dtype=mstype.uint8))
        
        if compress_statistics:
            qlora_layer.scale.set_data(Tensor(scale[0], dtype=mstype.int8))
            qlora_layer.scale_scale.set_data(Tensor(scale[1], dtype=mstype.float32))
        else:
            qlora_layer.scale.set_data(Tensor(scale, dtype=mstype.float32))
        
        qlora_layer.offset.set_data(Tensor(offset, dtype=mstype.float32))
        
        if has_bias:
            qlora_layer.bias.set_data(linear_layer.bias.data)
        
        print(f"Converted Linear({in_features}, {out_features}) to Linear4bitWithLoRA")
        print(f"  - Group size: {group_size}")
        print(f"  - Double quantization: {compress_statistics}")
        print(f"  - LoRA rank: {r}")
        print(f"  - LoRA alpha: {lora_alpha}")
        print(f"  - Memory saved: ~75%")
        print(f"  - Trainable parameters: {2 * in_features * r + r * out_features:,}")
        
        return qlora_layer
    
    def get_trainable_params(self):
        """
        Get trainable parameters
        
        Returns:
            list: List of trainable parameters
        """
        trainable_params = []
        for name, param in self.parameters_and_names():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params
    
    def print_trainable_params(self):
        """Print trainable parameter information"""
        trainable_params = self.get_trainable_params()
        total_params = sum(p.size for _, p in trainable_params)
        
        print("=" * 60)
        print("Trainable Parameters")
        print("=" * 60)
        for name, param in trainable_params:
            print(f"{name:<30} {param.shape} ({param.size:,} params)")
        print("-" * 60)
        print(f"Total: {total_params:,} trainable parameters")
        print("=" * 60)


class Linear8bitWithLoRA(nn.Cell if MINDSPORE_AVAILABLE else object):
    """
    8-bit quantization + LoRA layer
    
    Combines INT8 quantization with LoRA adapter.
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        bias (bool): Whether to use bias. Default: True
        r (int): LoRA rank. Default: 8
        lora_alpha (int): LoRA scaling factor. Default: 16
        lora_dropout (float): LoRA dropout probability. Default: 0.0
    
    Examples:
        >>> layer = Linear8bitWithLoRA(768, 3072, r=8, lora_alpha=16)
        >>> x = Tensor(np.random.randn(32, 768), dtype=ms.float32)
        >>> out = layer(x)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0
    ):
        if not MINDSPORE_AVAILABLE:
            raise ImportError("MindSpore is required for Linear8bitWithLoRA")
        
        super().__init__()
        
        from .linear import Linear8bit
        
        # INT8 quantized layer
        self.base_layer = Linear8bit(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            has_fp16_weights=False
        )
        
        # Freeze quantized weights
        self.base_layer.weight.requires_grad = False
        self.base_layer.scale.requires_grad = False
        if self.base_layer.offset is not None:
            self.base_layer.offset.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        
        # LoRA adapter
        self.lora = LoRALinear(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    
    def construct(self, x):
        """Forward pass"""
        # Main path: INT8 quantized computation
        out_main = self.base_layer(x)
        
        # LoRA path
        out_lora = self.lora(x)
        
        # Merge outputs
        return out_main + out_lora


def freeze_model_except_lora(model):
    """
    Freeze all parameters in model except LoRA parameters
    
    Args:
        model: Model
    
    Returns:
        tuple: (frozen_count, trainable_count)
    
    Examples:
        >>> frozen, trainable = freeze_model_except_lora(model)
        >>> print(f"Frozen parameters: {frozen}, Trainable parameters: {trainable}")
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required")
    
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.parameters_and_names():
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_count += 1
        else:
            param.requires_grad = False
            frozen_count += 1
    
    return frozen_count, trainable_count


def print_lora_info(model):
    """
    Print information about LoRA parameters in model
    
    Args:
        model: Model
    
    Examples:
        >>> print_lora_info(model)
    """
    if not MINDSPORE_AVAILABLE:
        raise ImportError("MindSpore is required")
    
    lora_params = []
    total_lora_params = 0
    
    for name, param in model.parameters_and_names():
        if 'lora' in name.lower():
            lora_params.append((name, param))
            total_lora_params += param.size
    
    if len(lora_params) == 0:
        print("No LoRA parameters found in model")
        return
    
    print("=" * 60)
    print("LoRA Parameter Information")
    print("=" * 60)
    for name, param in lora_params:
        print(f"{name:<40} {param.shape} ({param.size:,})")
    print("-" * 60)
    print(f"Total: {total_lora_params:,} LoRA parameters")
    print("=" * 60)
