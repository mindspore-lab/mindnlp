# SmoothQuant

SmoothQuant is a post-training quantization method for large language models.

## Overview

SmoothQuant provides:

- **Activation smoothing**: Migrates quantization difficulty from activations to weights
- **INT8 inference**: Enables efficient INT8 inference without significant accuracy loss
- **Easy integration**: Works with existing models

## Method

SmoothQuant smooths the activation outliers by migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation.

```
Y = (Xdiag(s)^-1) · (diag(s)W) = X̂Ŵ
```

where `s` is a smoothing factor that balances the quantization difficulty.

## Usage

```python
from mindnlp.quant.smooth_quant import SmoothQuantConfig

# Configure SmoothQuant
config = SmoothQuantConfig(
    smooth_alpha=0.5,  # Smoothing factor
    calibration_samples=512
)

# Apply to model
# ... (implementation depends on model type)
```

## Parameters

- **smooth_alpha**: Controls how much difficulty is migrated from activations to weights (0.0-1.0)
- **calibration_samples**: Number of samples for calibration

## References

- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)

## Notes

- Requires calibration data for optimal results
- Works best with transformer models
- Achieves INT8 quantization with minimal accuracy degradation
