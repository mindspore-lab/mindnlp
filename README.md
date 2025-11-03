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

## Table of Contents

- [ MindNLP](#-mindnlp)
  - [Table of Contents](#table-of-contents)
  - [News 📢](#news-)
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

## News 📢

* ⚡ **MindNLP Core support Pytorch compatible:** To meet ecosystem compatibility requirements, we provide the `mindnlp.core` module to support compatibility with PyTorch interfaces. This module is built upon MindSpore's foundational APIs and operators, enabling model development using syntax similar to PyTorch. It also supports taking over torch interfaces through a Proxy, allowing the use of MindSpore for acceleration on Ascend hardware without the need for code modifications. The specific usage is as follows:

    ```python
    import mindnlp # import mindnlp lib will enable proxy automaticlly
    import torch
    from torch import nn

    # all torch.xx apis will be mapped to mindnlp.core.xx
    net = nn.Linear(10, 5)
    x = torch.randn(3, 10)
    out = net(x)
    print(out.shape)
    # core.Size([3, 5])
    ```

    It is particularly noteworthy that MindNLP supports several features not yet available in MindSpore, which enables better support for model serialization, heterogeneous computing, and other scenarios:
    1. ​Dispatch Mechanism Support: Operators are dispatched to the appropriate backend based on Tensor.device.
    2. ​Meta Device Support: Allows for shape inference without performing actual computations.
    3. ​Numpy as CPU Backend: Supports using NumPy as a CPU backend for acceleration.
    4. ​Tensor.to for Heterogeneous Data Movement: Facilitates the movement of data across different devices using `Tensor.to`.

* 🔥 **Fully compatible with 🤗HuggingFace:** It enables seamless execution of any Transformers/Diffusers models on MindSpore across all hardware platforms (GPU/Ascend/CPU).
    
    You may still invoke models through MindNLP as shown in the example code below:

    ```python
    from mindnlp.transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    inputs = tokenizer("Hello world!", return_tensors='ms')
    outputs = model(**inputs)
    ```

    You can also directly use the native HuggingFace library(like transformers, diffusers, etc.) via the following approach as demonstrated in the example code:

    - For huggingface transformers:

    ```python
    import mindspore
    import mindnlp
    from transformers import pipeline

    chat = [
        {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]

    pipeline = pipeline(task="text-generation", model="Qwen/Qwen3-8B", ms_dtype=mindspore.bfloat16, device_map="auto")
    response = pipeline(chat, max_new_tokens=512)
    print(response[0]["generated_text"][-1]["content"])
    ```

    - For huggingface diffuers:

    ```python
    import mindspore
    import mindnlp
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", ms_dtype=mindspore.float16, device_map='cuda')
    pipeline("An image of a squirrel in Picasso style").images[0]
    ```

Notice ⚠️: Due to differences in autograd and parallel execution mechanisms, any training or distributed execution code must utilize the interfaces provided by MindNLP.

## Installation

#### Install from Pypi

You can install the official version of MindNLP which is uploaded to pypi.

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
| 0.5.0           | >=2.5.0, <=2.7.0  | >=3.10, <=3.11           |
| 0.6.x           | >=2.7.1.            | >=3.10, <=3.11           |

## Introduction

MindNLP is an open source NLP library based on MindSpore. It supports a platform for solving natural language processing tasks, containing many common approaches in NLP. It can help researchers and developers to construct and train models more conveniently and rapidly.

The master branch works with **MindSpore master**.

#### Major Features

- **Comprehensive data processing**: Several classical NLP datasets are packaged into friendly module for easy use, such as Multi30k, SQuAD, CoNLL, etc.
- **Friendly NLP model toolset**: MindNLP provides various configurable components. It is friendly to customize models using MindNLP.
- **Easy-to-use engine**: MindNLP simplified the complicated training process in MindSpore. It supports Trainer and Evaluator interfaces to train and evaluate models easily.


## Supported models

Since there are too many supported models, please check [here](https://mindnlp.cqu.ai/supported_models)

<!-- ## Tutorials

- (list of more tutorials...) -->

<!-- ## Notes -->

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

MindSpore is an open source project that welcomes any contribution and feedback.  
We wish that the toolbox and benchmark could serve the growing research  
community by providing a flexible as well as standardized toolkit to re-implement existing methods  
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{mindnlp2022,
    title={{MindNLP}: Easy-to-use and high-performance NLP and LLM framework based on MindSpore},
    author={MindNLP Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindnlp}},
    year={2022}
}
```
