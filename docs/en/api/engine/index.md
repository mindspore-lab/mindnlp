# Engine

MindNLP's Engine is a core component specially designed for the training model. It encapsulates the entire process of the model training and provides a series of advanced functions, making the training process more convenient and efficient.

## Introduction

Engine can automatically manage the iterative process of the model on the training dataset and make it more convenient to record the loss, indicators and other information during the training process. It greatly simplifies the complexity of model training and improves the development efficiency.

## Supported Algorithms

| Algorithm                                 | Description                                                                |
|-------------------------------------------|----------------------------------------------------------------------------|
| [trainer](./trainer/base.md)              | Trainer is a simple but feature-complete training and eval loop in Engine. |
| [callbacks](./callbacks.md)               | Callbacks to use with the Trainer class and customize the training loop.   |
| [export](./export.md)                     | Export models to other IR format.                                          |
| [utils](./utils.md)                       | Utils for engine.                                                          |

