import argparse
import os

import mindspore
from mindspore.nn import AdamWeightDecay
from mindnlp import load_dataset, process
from tqdm import tqdm
# from mindnlp.metrics import 

from mindnlp.peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftType,
)
from mindnlp.dataset import MRPC, MRPC_Process

from mindnlp.transforms import RobertaTokenizer
from mindnlp.models import RobertaConfig, RobertaForSequenceClassification


batch_size = 32
model_name_or_path = "roberta-base"
task = "mrpc"
peft_type = PeftType.LORA
device = "GPU" # "cuda"
num_epochs = 20
lr = 3e-4
warmup_ratio = 0.06

mrpc_train, mrpc_test = MRPC()


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
cols = ['sentence1', 'sentence2']
def process_dataset(dataset, tokenizer, column_names, batch_size, max_seq_len=512, shuffle=False):
    # tokenize
    for col in column_names:
        dataset = dataset.map(tokenizer, input_columns=col)

    return dataset

ds = process_dataset(mrpc_train, tokenizer, column_names=cols, batch_size=batch_size)

model_config = RobertaConfig(num_labels=2)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=model_config )
print(model)

peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

print(peft_model)
optimizer = AdamWeightDecay(params=peft_model.trainable_params(), learning_rate=lr)
