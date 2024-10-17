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


### News 📢

* 🔥 **Latest Features**

  * 🤗 **250+** Pretrained models support ***huggingface transformers-like apis***.
    You can use them easily by following code snippet:
    ```python
    from mindnlp.transformers import AutoModel

    model = AutoModel.from_pretrained('bert-base-cased')
    ```
  * **Full Platform Support**: Comprehensive support for `Ascend 910 series`, `Ascend 310B (Orange Pi)`, `GPU`, and `CPU`. (Note: Currently the only AI development kit available on Orange Pi.)
  * **Distributed Parallel Inference**: Multi-device, multi-process parallel inference support for models exceeding 10B parameters.
  * **Quantization Algorithm Support**: SmoothQuant available for Orange Pi; bitsandbytes-like int8 quantization supported on GPU.
  * **Sentence Transformer Support**: Enables efficient RAG (Retrieval-Augmented Generation) development.
  * **Dynamic Graph Performance Optimization**: Achieves PyTorch+GPU-level inference speeds for dynamic graphs on Ascend hardware (tested Llama performance at **85ms/token**).
  * **True Static and Dynamic Graph Unification**: One-line switching to graph mode with `mindspore.jit`, fully compatible with ***Hugging Face code style*** for both ease of use and rapid performance improvement. Tested Llama performance on Ascend hardware reaches 2x dynamic graph speed (**45ms/token**), consistent with other MindSpore static graph-based suites.
  * **Extensive LLM Application Updates**: Includes `Text information extraction`, `Chatbots`, `Speech recognition`, `ChatPDF`, `Music generation`, `Code generation`, `Voice clone`, etc. With increased model support, even more exciting applications await development!


### Installation

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
| 0.4.x           | >=2.2.x           | >=3.9, <=3.11            |

### Introduction

MindNLP is an open source NLP library based on MindSpore. It supports a platform for solving natural language processing tasks, containing many common approaches in NLP. It can help researchers and developers to construct and train models more conveniently and rapidly.

The master branch works with **MindSpore master**.

#### Major Features

- **Comprehensive data processing**: Several classical NLP datasets are packaged into friendly module for easy use, such as Multi30k, SQuAD, CoNLL, etc.
- **Friendly NLP model toolset**: MindNLP provides various configurable components. It is friendly to customize models using MindNLP.
- **Easy-to-use engine**: MindNLP simplified the complicated training process in MindSpore. It supports Trainer and Evaluator interfaces to train and evaluate models easily.


### Supported models

Since there are too many supported models, please check [here](https://mindnlp.cqu.ai/supported_models)

<!-- ## Tutorials

- (list of more tutorials...) -->

<!-- ## Notes -->

### License

This project is released under the [Apache 2.0 license](LICENSE).

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Github Issues](https://github.com/mindspore-lab/mindnlp/issues).

### Acknowledgement

MindSpore is an open source project that welcomes any contribution and feedback.  
We wish that the toolbox and benchmark could serve the growing research  
community by providing a flexible as well as standardized toolkit to re-implement existing methods  
and develop their own new semantic segmentation methods.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{mindnlp2022,
    title={{MindNLP}: Easy-to-use and high-performance NLP and LLM framework based on MindSpore},
    author={MindNLP Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindnlp}},
    year={2022}
}
```
