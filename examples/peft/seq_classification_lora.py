import argparse
import os

import mindspore
from mindspore import nn
from mindspore.nn import AdamWeightDecay
from mindnlp import load_dataset, process
from tqdm import tqdm

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

optimizer = AdamWeightDecay(params=peft_model.trainable_params(), learning_rate=lr)
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, criterion, train_dataloader, eval_dataloader, epochs):
    def forward_fn(data, label):
        logits = model(data)
        loss = criterion(logits, label)
        return loss, logits

    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_one_epoch():
        model.set_train(True)
        total_loss, total_step = 0, 0
        num_batches = len(train_dataloader)

        with tqdm(total=num_batches) as t:
            for xb, yb in train_dataloader:
                # forward + grad
                (loss, logits), grad = grad_fn(xb, yb)
                # update model params
                optimizer(grad)
                total_loss += loss.asnumpy()
                total_step += 1
                curr_loss = total_loss / total_step  # 当前的平均loss
                t.set_postfix({'train-loss': f'{curr_loss:.2f}'})
            
        return total_loss / total_step
    
    def eval_one_epoch():
        model.set_train(False)
        total_loss, total_step = 0, 0
        num_batches = len(eval_dataloader)
        with tqdm(total=num_batches) as t:
            for xb, yb in eval_dataloader:
                (loss, logits), grad = grad_fn(xb, yb)
                total_loss += loss.asnumpy()
                total_step += 1
                curr_loss = total_loss / total_step  # 当前的平均loss
                t.set_postfix({'eval-loss': f'{curr_loss:.2f}'})

        return total_loss / total_step

    # train start from here
    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch()
        eval_loss = eval_one_epoch()
        # logging
        print(f"epoch:{epoch} train_loss:{train_loss} eval_loss:{eval_loss}")
        
