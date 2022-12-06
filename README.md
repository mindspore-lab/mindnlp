# <center> MindNLP

<p align="center">
    <a href="https://mindnlp.cqu.ai/en/latest/">
        <img alt="docs" src="https://img.shields.io/badge/docs-latest-blue">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindnlp/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/mindspore-ecosystem/mindnlp.svg">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindnlp/pulls">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-pink.svg">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindnlp/issues">
        <img alt="open issues" src="https://img.shields.io/github/issues/mindspore-ecosystem/mindnlp">
    </a>
    <a href="https://github.com/mindspore-ecosystem/mindnlp/actions">
        <img alt="ci" src="https://github.com/mindspore-ecosystem/mindnlp/actions/workflows/ut_test.yaml/badge.svg">
    </a>
</p>

[Introduction](#introduction) |
[Quick Links](#quick-links) |
[Installation](#installation) |
[Get Started](#get-started) |
[Tutorials](#tutorials) |
[Notes](#notes)

## Introduction

MindNLP is an open source NLP library based on MindSpore. It supports a platform for solving natural language processing tasks, containing many common approaches in NLP. It can help researchers and developers to construct and train models more conveniently and rapidly.

The master branch works with **MindSpore master**.

### Major Features

- **Comprehensive data processing**: Several classical NLP datasets are packaged into friendly module for easy use, such as Multi30k, SQuAD, CoNLL, etc.
- **Friendly NLP model toolset**: MindNLP provides various configurable components. It is friendly to customize models using MindNLP.
- **Easy-to-use engine**: MindNLP simplified complicated training process in MindSpore. It supports Trainer and Evaluator interfaces to train and evaluate models easily.

## Quick Links

- [Website](https://mindnlp.cqu.ai/en/latest/)
- [Examples](https://github.com/mindspore-ecosystem/mindnlp/tree/master/examples)
- ...

## Installation

### Dependency

- mindspore >= ...
- ...

### Install from source

To install MindNLP from source, please run:

`pip install git+https://github.com/mindspore-ecosystem/mindnlp.git`

## Get Started

We will next quickly implement a sentiment classification task by using mindnlp.

### Define Model

```python
import math
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Uniform, HeUniform
from mindnlp.abc import Seq2vecModel

class Head(nn.Cell):
    """
    Head for Sentiment Classification model
    """
    def __init__(self, hidden_dim, output_dim, dropout):
        super().__init__()
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc = nn.Dense(hidden_dim * 2, output_dim, weight_init=weight_init, bias_init=bias_init)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, context):
        context = ops.concat((context[-2, :, :], context[-1, :, :]), axis=1)
        context = self.dropout(context)
        return self.sigmoid(self.fc(context))


class SentimentClassification(Seq2vecModel):
    """
    Sentiment Classification model
    """
    def __init__(self, encoder, head):
        super().__init__(encoder, head)
        self.encoder = encoder
        self.head = head

    def construct(self, text):
        _, (hidden, _), _ = self.encoder(text)
        output = self.head(hidden)
        return output
```
    
### Define Hyperparameters
The following are some of the required hyperparameters in the model training process.
```python
# define Models & Loss & Optimizer
hidden_size = 256
output_size = 1
num_layers = 2
bidirectional = True
drop = 0.5
lr = 0.001
```

### Data Preprocessing
The dataset was downloaded and preprocessed by calling the interface of dataset in mindnlp.

Load dataset:
```python
from mindnlp.dataset import load

imdb_train, imdb_test = load('imdb', shuffle=True)
```

Initializes the vocab and tokenizer for preprocessing:
```python
from mindnlp.modules import Glove
from mindnlp.dataset.transforms import BasicTokenizer

embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)
tokenizer = BasicTokenizer(True)
```

The loaded dataset is preprocessed and divided into training and validation:
```python
from mindnlp.dataset import process

imdb_train = process('imdb', imdb_train, tokenizer=tokenizer, vocab=vocab, \
                     bucket_boundaries=[400, 500], max_len=600, drop_remainder=True)
imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])
```

### Instantiate Model
```python
from mindnlp.modules import RNNEncoder

lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                     dropout=drop, bidirectional=bidirectional)
sentiment_encoder = RNNEncoder(embedding, lstm_layer)
sentiment_head = Head(hidden_size, output_size, drop)
net = SentimentClassification(sentiment_encoder, sentiment_head)
```

### Training Process
Now that we have completed all the preparations, we can begin to train the model.
```python
from mindnlp.engine.metrics import Accuracy
from mindnlp.engine.trainer import Trainer

# define metrics
metric = Accuracy()

# define trainer
trainer = Trainer(network=net, train_dataset=imdb_train, eval_dataset=imdb_valid, metrics=metric,
                  epochs=5, loss_fn=loss, optimizer=optimizer)
trainer.run(tgt_columns="label", jit=False)
print("end train")
```

## Tutorials

- (list of more tutorials...)

## Notes

### License

This project is released under the [Apache 2.0 license](LICENSE).

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Gitee Issues](https://gitee.com/mindspore/text/issues).

### Acknowledgement

MindSpore is an open source project that welcome any contribution and feedback.  
We wish that the toolbox and benchmark could serve the growing research  
community by providing a flexible as well as standardized toolkit to reimplement existing methods  
and develop their own new semantic segmentation methods.

### Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{mindnlp2022,
    title={{MindNLP}: a MindSpore NLP library},
    author={MindNLP Contributors},
    howpublished = {\url{https://github.com/mindlab-ai/mindnlp}},
    year={2022}
}
```
