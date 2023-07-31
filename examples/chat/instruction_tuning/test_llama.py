import os

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn

from mindnlp import load_dataset
from mindnlp.transforms import PadTransform, GPTTokenizer

from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy


from datasets import load_dataset as hf_load

from mindnlp.models.llama import LlamaConfig, LlamaForCausalLM


## eval mmlu



dataset_test = load_dataset('/home/cjl/cjldatasets/CMMLU/data', splits='test')
metric = Accuracy()

model = LlamaForCausalLM.from_pretrained('mindnlp/llama-base')

evaluator = Evaluator(
    network=model, 
    eval_dataset=dataset_test, 
    metrics=metric
)

evaluator.run()