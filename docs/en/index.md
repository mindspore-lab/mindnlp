---
hide:
  - navigation
---
# <center> MindNLP

<p align="center">
    <a href="https://mindnlp.cqu.ai/en/latest/">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/mindspore-lab/mindnlp/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-lab/mindnlp.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindnlp/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/mindspore-lab/mindnlp/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/mindspore-lab/mindnlp">
    </a>
    <a href="https://github.com/mindspore-lab/mindnlp/actions">
        <img alt="ci" src="https://github.com/mindspore-lab/mindnlp/actions/workflows/ci_pipeline.yaml/badge.svg">
    </a>
</p>

**MindNLP** stands for **MindSpore + Natural Language Processing**, representing seamless compatibility with the HuggingFace ecosystem. MindNLP lets you use the rich HuggingFace model hub together with MindSpore acceleration.

## Features ✨

### 1. 🤗 Full HuggingFace Compatibility
Run Transformers/Diffusers models on MindSpore (GPU/Ascend/CPU) without code changes.

**Transformers example**
```python
import mindspore
import mindnlp
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipe = pipeline(task="text-generation", model="Qwen/Qwen3-8B",
                ms_dtype=mindspore.bfloat16, device_map="auto")
resp = pipe(chat, max_new_tokens=512)
print(resp[0]["generated_text"][-1]["content"])
```

**Diffusers example**
```python
import mindspore
import mindnlp
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ms_dtype=mindspore.float16,
    device_map="cuda",
)
pipe("An image of a squirrel in Picasso style").images[0]
```

**MindNLP native interface**
```python
from mindnlp.transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
inputs = tokenizer("Hello world!", return_tensors="ms")
outputs = model(**inputs)
```
> Training or distributed execution should use MindNLP APIs due to autograd/parallel differences.

### 2. ⚡ High-Performance Features Powered by MindSpore
- PyTorch-compatible API via `mindtorch` (`mindnlp.core`), mapping `torch.*` to MindSpore-backed kernels.
- Dispatch based on `Tensor.device`, meta device support, NumPy CPU backend, and enhanced `Tensor.to()` for heterogeneous movement.
- Better support for serialization, heterogeneous computing, and complex deployment.

### 3. 🌐 Broad Model & Task Coverage
- LLMs such as Llama, GLM, RWKV, Qwen, etc.; examples for pretrain/finetune/inference.
- 60+ pretrained models with transformers-like APIs; see examples and supported models list.

## Installation

#### Install from Pypi

You can install the official version of MindNLP which uploaded to pypi.

```bash
pip install mindnlp
```

#### Daily build

You can download MindNLP daily wheel from [here](https://repo.mindspore.cn/mindspore-lab/mindnlp/newest/any/).

#### Install from source

To install MindNLP from source, please run:

```bash
pip install git+https://github.com/mindspore-lab/mindnlp.git
# or
git clone https://github.com/mindspore-lab/mindnlp.git
cd mindnlp
bash scripts/build_and_reinstall.sh
```

#### Version Compatibility

| MindNLP version | MindSpore version | Supported Python version |
|-----------------|-------------------|--------------------------|
| master          | daily build       | >=3.7.5, <=3.9           |
| 0.1.1           | >=1.8.1, <=2.0.0  | >=3.7.5, <=3.9           |
| 0.2.x           | >=2.1.0           | >=3.8, <=3.9             |
| 0.3.x           | >=2.1.0, <=2.3.1  | >=3.8, <=3.9             |
| 0.4.x           | >=2.2.x, <=2.5.0  | >=3.9, <=3.11            |
| 0.5.x           | >=2.5.0, <=2.7.0  | >=3.10, <=3.11           |
| 0.6.x           | >=2.7.1            | >=3.10, <=3.11           |

## Supported models

Since there are too many supported models, please check [here](https://mindnlp.cqu.ai/supported_models)

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Github Issues](https://github.com/mindspore-lab/mindnlp/issues).

## MindSpore NLP SIG

MindSpore NLP SIG (Natural Language Processing Special Interest Group) is the main development team of the MindNLP framework. It aims to collaborate with developers from both industry and academia who are interested in research, application development, and the practical implementation of natural language processing. Our goal is to create the best NLP framework based on the domestic framework MindSpore. Additionally, we regularly hold NLP technology sharing sessions and offline events. Interested developers can join our SIG group using the QR code below.

<div align="center">
    <img src="./assets/qrcode_qq_group.jpg" width="250" />
</div>

## Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback.  
We wish that the toolbox and benchmark could serve the growing research  
community by providing a flexible as well as standardized toolkit to re-implement existing methods  
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{mindnlp2022,
    title={{MindNLP}: Easy-to-use and high-performance NLP and LLM framework based on MindSpore},
    author={MindNLP Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindnlp}},
    year={2022}
}
```
