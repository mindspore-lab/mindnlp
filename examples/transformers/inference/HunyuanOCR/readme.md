---
license: other
language:
- zh
- en
pipeline_tag: image-text-to-text
library_name: transformers
---

<p align="center">
 <img src="https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/assets/hyocr-head-img.png?raw=true" width="80%"/> <br>
</p>


<p align="center">
<a href="https://huggingface.co/spaces/tencent/HunyuanOCR"><b>🎯 Demo</b></a> |
<a href="https://huggingface.co/tencent/HunyuanOCR"><b>📥 Model Download</b></a> |
<a href="https://arxiv.org/abs/2511.19575"><b>📄 Technical Report</b></a> |
<a href="https://github.com/Tencent-Hunyuan/HunyuanOCR"><b>🌟 Github</b></a>
</p>

<h2>
<p align="center">
  <a href="https://arxiv.org/abs/2511.19575">HunyuanOCR</a>
</p>
</h2>


## Notice

The official repo of [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) do not support official `transformers` and only provide a commit version to use. We modify the official implementation as `remote_code` to support the official `transformers` version. You can use [HunyuanOCR] with the latest version of `transformers` easily.

## 📖 Introduction
**HunyuanOCR** stands as a leading end-to-end OCR expert VLM powered by Hunyuan's native multimodal architecture. With a remarkably lightweight 1B parameter design, it has achieved multiple state-of-the-art benchmarks across the industry. The model demonstrates mastery in **complex multilingual document parsing** while excelling in practical applications including **text spotting, open-field information extraction, video subtitle extraction, and photo translation**.


## 🚀 Quick Start with Transformers

### Installation

```bash
pip install transformers==4.57.3
pip install git+https://github.com/mindspore-lab/mindnlp
```


### Model Inference with MindSpore + MindNLP

```python
import mindtorch
import mindnlp
from transformers import AutoProcessor
from transformers import AutoModel
from PIL import Image

def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n<8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text

model_name_or_path = "lvyufeng/HunyuanOCR"
processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
img_path = "image_ocr.jpg"
image_inputs = Image.open(img_path)
messages1 = [
    {"role": "system", "content": ""},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": (
                "检测并识别图片中的文字，将文本坐标格式化输出。"
            )},
        ],
    }
]
messages = [messages1]
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

inputs = processor(
    text=texts,
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
model = AutoModel.from_pretrained(
    model_name_or_path,
    attn_implementation="eager",
    dtype=mindtorch.float16,
    device_map="auto",
    trust_remote_code=True
)
with mindtorch.no_grad():
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=16384, do_sample=False)
if "input_ids" in inputs:
    input_ids = inputs.input_ids
else:
    print("inputs: # fallback", inputs)
    input_ids = inputs.inputs
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
]
output_texts = clean_repeated_substrings(processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
))
print(output_texts)

```

## 💬 Application-oriented Prompts

| Task | English | Chinese |
|------|---------|---------|
| **Spotting** | Detect and recognize text in the image, and output the text coordinates in a formatted manner. | 检测并识别图片中的文字，将文本坐标格式化输出。 |
| **Parsing** | • Identify the formula in the image and represent it using LaTeX format.<br><br>• Parse the table in the image into HTML.<br><br>• Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.<br><br>• Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. | • 识别图片中的公式，用 LaTeX 格式表示。<br><br>• 把图中的表格解析为 HTML。<br><br>• 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。<br><br>• 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。 |
| **Information Extraction** | • Output the value of Key.<br><br>• Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format.<br><br>• Extract the subtitles from the image. | • 输出 Key 的值。<br><br>• 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。<br><br>• 提取图片中的字幕。 |
| **Translation** | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. | 先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 |


## 📚 Citation
```
@misc{hunyuanvisionteam2025hunyuanocrtechnicalreport,
      title={HunyuanOCR Technical Report}, 
      author={Hunyuan Vision Team and Pengyuan Lyu and Xingyu Wan and Gengluo Li and Shangpin Peng and Weinong Wang and Liang Wu and Huawen Shen and Yu Zhou and Canhui Tang and Qi Yang and Qiming Peng and Bin Luo and Hower Yang and Xinsong Zhang and Jinnian Zhang and Houwen Peng and Hongming Yang and Senhao Xie and Longsha Zhou and Ge Pei and Binghong Wu and Kan Wu and Jieneng Yang and Bochao Wang and Kai Liu and Jianchen Zhu and Jie Jiang and Linus and Han Hu and Chengquan Zhang},
      year={2025},
      journal={arXiv preprint arXiv:2511.19575},
      url={https://arxiv.org/abs/2511.19575}, 
}
```

## 🙏 Acknowledgements
We would like to thank [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [MinerU](https://github.com/opendatalab/MinerU), [MonkeyOCR](https://github.com/Yuliang-Liu/MonkeyOCR), [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), [dots.ocr](https://github.com/rednote-hilab/dots.ocr) for their valuable models and ideas.
We also appreciate the benchmarks: [OminiDocBench](https://github.com/opendatalab/OmniDocBench), [OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR/tree/main/OCRBench), [DoTA](https://github.com/liangyupu/DIMTDA).

