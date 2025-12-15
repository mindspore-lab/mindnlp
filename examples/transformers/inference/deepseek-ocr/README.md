---
pipeline_tag: image-text-to-text
language:
- multilingual
tags:
- mindspore
- mindnlp
- deepseek
- vision-language
- ocr
- custom_code
license: mit
---
<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek AI" />
</div>
<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" />
  </a>
  <a href="https://huggingface.co/lvyufeng/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>




<p align="center">
  <a href="https://github.com/mindspore-lab/mindnlp/tree/master/examples/transformers/inference/deepseek-ocr"><b>🌟 Github</b></a> |
  <a href="https://huggingface.co/lvyufeng/DeepSeek-OCR"><b>📥 Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 Paper Link</b></a> |
  <a href=""><b>📄 Arxiv Paper Link</b></a> |
</p>
<h2>
<p align="center">
  DeepSeek-OCR: Contexts Optical Compression
</p>
</h2>
<p align="center">
  Explore the boundaries of visual-text compression.
</p>

## Usage

This application now supports running on both Ascend 910 and OrangePi AIpro. Feel free to give it a try!

### Environment Prerequisites

Install the required dependencies first:

```
mindspore==2.7.0
mindnlp==0.5.0rc4
transformers==4.57.1
tokenizers
einops
addict
easydict
```

### Code Implementation

Core inference code for DeepSeek-OCR implemented by MindSpore NLP:

```python
import os
import mindnlp
import torch
from transformers import AutoModel, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = 'lvyufeng/DeepSeek-OCR-Community-Latest'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='sdpa', trust_remote_code=True, use_safetensors=True, device_map='auto')
model = model.eval()

# Configure inference parameters
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# prompt = "<image>\nFree OCR. "  # Feel free to try other prompt templates
image_file = 'your_image.jpg'  # Replace with your image path
output_path = 'your/output/dir'  # Replace with your output directory

"""
Inference function parameter explanation
----------------------------------------
Function signature:
infer(self, tokenizer, prompt='', image_file='', output_path='', base_size=1024, image_size=640, crop_mode=True, test_compress=False, save_results=False)

Parameter configurations for different model scales:
- Tiny: base_size=512, image_size=512, crop_mode=False
- Small: base_size=640, image_size=640, crop_mode=False
- Base: base_size=1024, image_size=1024, crop_mode=False
- Large: base_size=1280, image_size=1280, crop_mode=False
- Gundam: base_size=1024, image_size=640, crop_mode=True (default used here)
"""

# Run OCR inference with Gundam configuration
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size=1024, image_size=640, crop_mode=True, save_results = True, test_compress = True)
```

### How to Run

Execute the script to start OCR inference:

```bash
  python run_dpsk_ocr.py
```

### Outputs

After running the script, the model will generate two files in the specified output_path:
- OCR visualization result: `result_with_boxes.jpg` (image with text bounding boxes)
- Converted markdown file: `result.mmd` (structured text output from OCR)

## Acknowledgement

We would like to thank [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OneChart](https://github.com/LingyvKong/OneChart), [Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) for their valuable models and ideas.

We also appreciate the benchmarks: [Fox](https://github.com/ucaslcl/Fox), [OminiDocBench](https://github.com/opendatalab/OmniDocBench).
