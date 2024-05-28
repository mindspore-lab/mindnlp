# Dataset

MindNLP's dataset further encapsulates the Mindspore.dataset module and provides some more concise and easy-to-use APIs.

## Introduction:

MindNLP.dataset can provide APIs for loading and processing datasets (especially for NLP tasks), supporting loading datasets from the Hugging Face Hub or locally.

This module also provides a series of APIs for simple data operations (located in mindnlp.dataset.transformers), including `Truncate`, `AddToken`, `Lookup`, `PadTransform`, `BasicTokenizer`, and `JiebaTokenizer`.

At the same time, a base class for mapping functions is provided: `BaseMapFunction`, which is used to map or transform input data in some way.

## Category

| Category                                                     | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [load_dataset](https://github.com/mindspore-lab/mindnlp/blob/master/docs/en/api/dataset/load_dataset.md) | The main tool provided by the dataset for loading datasets from Hugging Face Hub or locally and converting them into a dataset format supported by MindSpore ([GeneratorDataset](https://www.mindspore.cn/docs/en/r2.3.0rc2/api_python/dataset/mindspore.dataset.GeneratorDataset.html)). |
| [BaseMapFunction](https://github.com/mindspore-lab/mindnlp/blob/master/docs/en/api/dataset/BaseMapFunction.md) | Base class for mapping functions that map input data to output data by specifying input and output data columns. |
| [transforms](https://github.com/mindspore-lab/mindnlp/blob/master/docs/en/api/dataset/transforms.md) | Data Processing Transformation Toolset                       |

