# <center> MindHF

<p align="center">
    <a href="https://mindhf.cqu.ai/en/latest/">
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

**MindHF** stands for **MindSpore + HuggingFace**, representing seamless compatibility with the HuggingFace ecosystem. The name also embodies **Harmonious & Fluid**, symbolizing our commitment to balancing compatibility with high performance. MindHF enables you to leverage the best of both worlds: the rich HuggingFace model ecosystem and MindSpore's powerful acceleration capabilities.

> **Note**: MindHF (formerly MindNLP) is the new name for this project. The `mindnlp` package name is still available for backward compatibility, but we recommend using `mindhf` going forward.

## Table of Contents

- [ MindHF](#-mindhf)
  - [Table of Contents](#table-of-contents)
  - [Features ✨](#features-)
  - [Installation](#installation)
      - [Install from Pypi](#install-from-pypi)
      - [Daily build](#daily-build)
      - [Install from source](#install-from-source)
      - [Version Compatibility](#version-compatibility)
  - [Introduction](#introduction)
      - [Major Features](#major-features)
  - [Supported models](#supported-models)
  - [License](#license)
  - [Feedbacks and Contact](#feedbacks-and-contact)
  - [MindSpore NLP SIG](#mindspore-nlp-sig)
  - [Acknowledgement](#acknowledgement)
  - [Citation](#citation)

## Features ✨

### 1. 🤗 Full HuggingFace Compatibility

MindHF provides seamless compatibility with the HuggingFace ecosystem, enabling you to run any Transformers/Diffusers models on MindSpore across all hardware platforms (GPU/Ascend/CPU) without code modifications.

#### Direct HuggingFace Library Usage

You can directly use native HuggingFace libraries (transformers, diffusers, etc.) with MindSpore acceleration:

**For HuggingFace Transformers:**

```python
import mindspore
import mindhf
from transformers import pipeline

chat = [
    {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
    {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
]

pipeline = pipeline(task="text-generation", model="Qwen/Qwen3-8B", ms_dtype=mindspore.bfloat16, device_map="auto")
response = pipeline(chat, max_new_tokens=512)
print(response[0]["generated_text"][-1]["content"])
```

**For HuggingFace Diffusers:**

```python
import mindspore
import mindhf
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", ms_dtype=mindspore.float16, device_map='cuda')
pipeline("An image of a squirrel in Picasso style").images[0]
```

#### MindHF Native Interface

You can also use MindHF's native interface for better integration:

```python
from mindhf.transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world!", return_tensors='ms')
outputs = model(**inputs)
```

> **Note**: Due to differences in autograd and parallel execution mechanisms, any training or distributed execution code must utilize the interfaces provided by MindHF.

### 2. ⚡ High-Performance Features Powered by MindSpore

MindHF leverages MindSpore's powerful capabilities to deliver exceptional performance and unique features:

#### PyTorch-Compatible API with MindSpore Acceleration

MindHF provides `mindtorch` (accessible via `mindhf.core`) for PyTorch-compatible interfaces, enabling seamless migration from PyTorch code while benefiting from MindSpore's acceleration on Ascend hardware:

```python
import mindhf  # Automatically enables proxy for torch APIs
import torch
from torch import nn

# All torch.xx APIs are automatically mapped to mindhf.core.xx (via mindtorch)
net = nn.Linear(10, 5)
x = torch.randn(3, 10)
out = net(x)
print(out.shape)  # core.Size([3, 5])
```

#### Advanced Features Beyond Standard MindSpore

MindHF extends MindSpore with several advanced features for better model development:

1. **Dispatch Mechanism**: Operators are automatically dispatched to the appropriate backend based on `Tensor.device`, enabling seamless multi-device execution.
2. **Meta Device Support**: Perform shape inference and memory planning without actual computations, significantly speeding up model development and debugging.
3. **NumPy as CPU Backend**: Use NumPy as a CPU backend for acceleration, providing better compatibility and performance on CPU devices.
4. **Heterogeneous Data Movement**: Enhanced `Tensor.to()` for efficient data movement across different devices (CPU/GPU/Ascend).

These features enable better support for model serialization, heterogeneous computing, and complex deployment scenarios.

## Installation

#### Install from Pypi

You can install the official version of MindHF which is uploaded to pypi.

```bash
pip install mindhf
```

> **Note**: The `mindnlp` package name is still available for backward compatibility, but we recommend using `mindhf` going forward.

#### Daily build

You can download MindHF daily wheel from [here](https://repo.mindspore.cn/mindspore-lab/mindhf/newest/any/).

#### Install from source

To install MindHF from source, please run:

```bash
pip install git+https://github.com/mindspore-lab/mindhf.git
# or
git clone https://github.com/mindspore-lab/mindhf.git
cd mindhf
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

| MindHF version | MindSpore version | Supported Python version |
|-----------------|-------------------|--------------------------|
| 0.6.x           | >=2.7.1.            | >=3.10, <=3.11           |


## Supported models

Since there are too many supported models, please check [here](https://mindhf.cqu.ai/supported_models)

<!-- ## Tutorials

- (list of more tutorials...) -->

<!-- ## Notes -->

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Github Issues](https://github.com/mindspore-lab/mindnlp/issues).


## MindSpore NLP SIG

MindSpore NLP SIG (Natural Language Processing Special Interest Group) is the main development team of the MindHF framework. It aims to collaborate with developers from both industry and academia who are interested in research, application development, and the practical implementation of natural language processing. Our goal is to create the best NLP framework based on the domestic framework MindSpore. Additionally, we regularly hold NLP technology sharing sessions and offline events. Interested developers can join our SIG group using the QR code below.

<div align="center">
    <img src="./assets/qrcode_qq_group.jpg" width="250" />
</div>


## Acknowledgement

MindSpore is an open source project that welcomes any contribution and feedback.  
We wish that the toolbox and benchmark could serve the growing research  
community by providing a flexible as well as standardized toolkit to re-implement existing methods  
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{mindhf2022,
    title={{MindHF}: Easy-to-use and high-performance NLP and LLM framework based on MindSpore},
    author={MindHF Contributors},
    howpublished = {\url{https://github.com/mindspore-lab/mindnlp}},
    year={2022},
    note={Formerly known as MindNLP}
}
```
