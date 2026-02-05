# Diffusers Patch

This module patches HuggingFace Diffusers to work with MindSpore as the backend.

## Overview

When you import `mindnlp`, it automatically patches the `diffusers` library to use MindSpore operations. This enables you to use all HuggingFace diffusion models directly.

## Usage

```python
import mindspore
import mindnlp  # Patches are applied automatically

from diffusers import DiffusionPipeline

# Load and run Stable Diffusion
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ms_dtype=mindspore.float16,
    device_map="cuda"
)

image = pipe("A beautiful sunset over mountains").images[0]
image.save("sunset.png")
```

## Supported Models

All diffusers models are supported, including:

- **Stable Diffusion**: v1.5, v2.x, XL
- **SDXL**: Stable Diffusion XL
- **ControlNet**: Conditional generation
- **Image-to-Image**: img2img pipelines
- **Inpainting**: Image inpainting models
- **LoRA**: Low-Rank Adaptation fine-tuning

## Example: Stable Diffusion XL

```python
import mindspore
import mindnlp
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    ms_dtype=mindspore.float16
)

image = pipe(
    prompt="A majestic lion in the savanna",
    num_inference_steps=30
).images[0]
```

## Example: ControlNet

```python
import mindspore
import mindnlp
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    ms_dtype=mindspore.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    ms_dtype=mindspore.float16
)
```

## Notes

- Image generation is primarily optimized for inference
- Training diffusion models may require additional setup
- GPU acceleration is recommended for reasonable generation speeds
