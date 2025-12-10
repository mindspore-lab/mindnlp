#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os

import mindspore
from mindspore.common.api import _no_grad

from tqdm import tqdm
from mindhf.core.optim import AdamW
from mindhf import evaluate
from mindhf.dataset import load_dataset
from mindhf.transformers import AutoModelForSequenceClassification, AutoTokenizer
from mindhf.transformers.optimization import get_linear_schedule_with_warmup
from mindhf.peft import (
    get_peft_model,
    PeftType,
    LoraConfig,
)

import faulthandler
faulthandler.enable()

# mindspore.set_context(pynative_synchronize=True)
# In[2]:

batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.PROMPT_TUNING
num_epochs = 20


# In[3]:


peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4


# In[4]:


if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# In[5]:


datasets = load_dataset("glue", task)
print(next(datasets['train'].create_dict_iterator()))


# In[6]:


from mindhf.dataset import BaseMapFunction

class MapFunc(BaseMapFunction):
    def __call__(self, sentence1, sentence2, label, idx):
        outputs = tokenizer(str(sentence1), str(sentence2), truncation=True, max_length=None)
        return outputs['input_ids'], outputs['attention_mask'], label


def get_dataset(dataset, tokenizer):
    input_colums=['sentence1', 'sentence2', 'label', 'idx']
    output_columns=['input_ids', 'attention_mask', 'labels']
    dataset = dataset.map(MapFunc(input_colums, output_columns),
                          input_colums, output_columns)
    dataset = dataset.padded_batch(batch_size, pad_info={'input_ids': (None, tokenizer.pad_token_id),
                                                         'attention_mask': (None, 0)})
    return dataset

train_dataset = get_dataset(datasets['train'], tokenizer)
eval_dataset = get_dataset(datasets['validation'], tokenizer)


# In[7]:


print(next(train_dataset.create_dict_iterator()))


# In[8]:


metric = evaluate.load("glue", task)


# In[9]:

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, attn_implementation='eager')
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# In[10]:


optimizer = AdamW(params=tuple(param for param in model.parameters() if param.requires_grad), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataset) * num_epochs),
    num_training_steps=(len(train_dataset) * num_epochs),
)


# In[ ]:


from mindhf.core import value_and_grad
def forward_fn(**batch):
    outputs = model(**batch)
    loss = outputs.loss
    return loss

grad_fn = value_and_grad(forward_fn, tuple(param for param in model.parameters() if param.requires_grad))

for epoch in range(num_epochs):
    model.set_train()
    train_total_size = train_dataset.get_dataset_size()
    for step, batch in enumerate(tqdm(train_dataset.create_dict_iterator(), total=train_total_size)):
        optimizer.zero_grad()
        loss = grad_fn(**batch)
        optimizer.step()
        lr_scheduler.step()

    model.set_train(False)
    eval_total_size = eval_dataset.get_dataset_size()
    for step, batch in enumerate(tqdm(eval_dataset.create_dict_iterator(), total=eval_total_size)):
        with _no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(axis=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)


# In[ ]:




