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

[Introduction](#introduction) |
[Quick Links](#quick-links) |
[Installation](#installation) |
[Get Started](#get-started) |
[Tutorials](#tutorials) |
[Notes](#notes)

## News 📢

* 🔥 **Latest Features**
  * 📃 Support PreTrained Models, including **[BERT](./mindnlp/models/bert)**, **[Roberta](./mindnlp/models/roberta)**, **[GPT2](./mindnlp/models/gpt2)** and **[T5](./mindnlp/models/t5)**.
    You can use them by following code snippet:
    ```python
    from mindnlp.models import BertModel

    model = BertModel.from_pretrained('bert-base-cased')
    ```



## Introduction

MindNLP is an open source NLP library based on MindSpore. It supports a platform for solving natural language processing tasks, containing many common approaches in NLP. It can help researchers and developers to construct and train models more conveniently and rapidly.

The master branch works with **MindSpore master**.

### Major Features

- **Comprehensive data processing**: Several classical NLP datasets are packaged into friendly module for easy use, such as Multi30k, SQuAD, CoNLL, etc.
- **Friendly NLP model toolset**: MindNLP provides various configurable components. It is friendly to customize models using MindNLP.
- **Easy-to-use engine**: MindNLP simplified complicated training process in MindSpore. It supports Trainer and Evaluator interfaces to train and evaluate models easily.

## Quick Links

- [Documentation](https://mindnlp.cqu.ai/en/latest/)
- [Examples](https://github.com/mindspore-ecosystem/mindnlp/tree/master/examples)
- ...

## Installation

### Dependency

- mindspore >= 1.8.1

### Install from source

To install MindNLP from source, please run:

```bash
pip install git+https://github.com/mindspore-ecosystem/mindnlp.git
```

## Get Started

We will next quickly implement a sentiment classification task by using mindnlp.

### Define Model

```python
import math
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Uniform, HeUniform
from mindnlp.abc import Seq2vecModel

class SentimentClassification(Seq2vecModel):
    def construct(self, text):
        _, (hidden, _), _ = self.encoder(text)
        context = ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        output = self.head(context)
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
from mindnlp.transforms import BasicTokenizer

embedding, vocab = Glove.from_pretrained('6B', 100, special_tokens=["<unk>", "<pad>"], dropout=drop)
tokenizer = BasicTokenizer(True)
```

The loaded dataset is preprocessed and divided into training and validation:
```python
from mindnlp.dataset import process

imdb_train = process('imdb', imdb_train, tokenizer=tokenizer, vocab=vocab, \
                     bucket_boundaries=[400, 500], max_len=600, drop_remainder=True)
imdb_test = process('imdb', imdb_test, tokenizer=tokenizer, vocab=vocab, \
                     bucket_boundaries=[400, 500], max_len=600, drop_remainder=False)
```

### Instantiate Model
```python
from mindnlp.modules import RNNEncoder

# build encoder
lstm_layer = nn.LSTM(100, hidden_size, num_layers=num_layers, batch_first=True,
                     dropout=dropout, bidirectional=bidirectional)
encoder = RNNEncoder(embedding, lstm_layer)

# build head
head = nn.SequentialCell([
    nn.Dropout(p=dropout),
    nn.Sigmoid(),
    nn.Dense(hidden_size * 2, output_size,
             weight_init=HeUniform(math.sqrt(5)),
             bias_init=Uniform(1 / math.sqrt(hidden_size * 2)))

])

# build network
network = SentimentClassification(encoder, head)
loss = nn.BCELoss(reduction='mean')
optimizer = nn.Adam(network.trainable_params(), learning_rate=lr)
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
trainer.run(tgt_columns="label")
print("end train")
```

<!-- ## Tutorials

- (list of more tutorials...) -->

<!-- ## Notes -->

### License

This project is released under the [Apache 2.0 license](LICENSE).

### Feedbacks and Contact

The dynamic version is still under development, if you find any issue or have an idea on new features, please don't hesitate to contact us via [Github Issues](https://github.com/mindspore-lab/mindnlp/issues).

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
