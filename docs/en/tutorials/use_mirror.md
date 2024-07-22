# Use mirror to download models and datasets

While the official Hugging Face repository offers numerous high-quality models and datasets, they may not be always accessible due to network issues. To make the access easier, MindNLP enables you to download models and datasets from a variety of huggingface mirrors or other model repositories.

Here we show you how to set your desired mirror.

You can either set the Hugging Face mirror through the environment variable, or more locally, specify the mirror in the `from_pretrained` method when downloading models.

## Set Hugging Face mirror through the environment variable

The Huggingface mirror used in MindNLP is controlled throught the `HF_ENDPOINT` environment variable.

You can either set this variable in the terminal before excuting your python script:
```bash
export HF_ENDPOINT="https://hf-mirror.com"
```
or set it within the python script using the `os` package:


```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

If the `HF_ENDPOINT` variable is not set explicitly by the user, MindNLP will use 'https://hf-mirror.com' by default. You can change this to the official Huggingface repository, 'https://huggingface.co'.

**Important:**

The URL should not include the last '/'. Setting the varialble to 'https://hf-mirror.com' will work, while setting it to 'https://hf-mirror.com/' will result in an error.

**Important:**

As the `HF_ENDPOINT` variable is read during the initial import of MindNLP, it is important to set the `HF_ENDPOINT` before importing MindNLP. If you are in a Jupyter Notebook, and MindNLP package is already imported, you may need to restart the notebook for the change to take effect.

Now you can download the model you want, for example:


```python
from mindnlp.transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
```

## Specify Hugging Face mirror in the `from_pretrained` method

Instead of setting the Hugging Face mirror globally through the environment variable, you can also specify the mirror for a single download operation in the `from_pretrained` method.

For example:


```python
from mindnlp.transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', mirror='modelscope', revision='master')
```

MindNLP accepts the following options for the `mirror` argument:

* 'huggingface'

    Download from the Hugging Face mirror specified through the `HF_ENDPOINT` environment variable. By default, it points to [HF-Mirror](https://hf-mirror.com).

* 'modelscope'

    Download from  [ModelScope](https://www.modelscope.cn).

* 'wisemodel'

    Download from [始智AI](https://www.wisemodel.cn).

* 'gitee'

    Dowload from the [Gitee AI Hugging Face repository](https://ai.gitee.com/huggingface).

* 'aifast'

    Download from [AI快站](https://aifasthub.com).

Note that not all models can be found from a single mirror, you may need to check whether the model you want to download is actually provided by the mirror you choose.

In addition to specifying the mirror, you also need to specify the `revision` argument. The `revision` argument can either be 'master' or 'main' depending on the mirror you choose. By default, `revision='main'`.

* If the `mirror` is 'huggingface', 'wisemodel' or 'gitee', set `revision='main'`.

* If the `mirror` is 'modelscope', set `revision='master'`.

* If the `mirror` is 'aifast', `revision` does not need to be specified.

