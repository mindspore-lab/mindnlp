#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mindspore
from mindhf.transformers import AutoModelForSeq2SeqLM
from mindhf.peft import get_peft_model, LoraConfig, TaskType
from mindhf.core import ops

from mindhf.transformers import AutoTokenizer
from mindhf.transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

checkpoint_name = "financial_sentiment_analysis_lora_v1.ckpt"
max_length = 128
lr = 1e-3
num_epochs = 3
batch_size = 8


# In[ ]:


# creating model
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


# In[ ]:


# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["label"]]},
    batched=True,
    num_proc=1,
)

dataset["train"][0]

# In[ ]:

print(dataset.source.ds)
classes = dataset.source.ds.features["label"].names
classes


# In[ ]:


train_dataset, validation_dataset = dataset.shuffle(64).split([0.9, 0.1])


# In[ ]:


def add_text_label(sentence, label):
    return sentence, label, classes[label.item()]

train_dataset = train_dataset.map(add_text_label, ['sentence', 'label'], ['sentence', 'label', 'text_label'])
validation_dataset = validation_dataset.map(add_text_label, ['sentence', 'label'], ['sentence', 'label', 'text_label'])


# In[ ]:


next(train_dataset.create_dict_iterator())


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


# In[ ]:


import numpy as np
from mindhf.dataset import BaseMapFunction
from threading import Lock
lock = Lock()

class MapFunc(BaseMapFunction):
    def __call__(self, sentence, label, text_label):
        lock.acquire()
        model_inputs = tokenizer(sentence, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(text_label, max_length=3, padding="max_length", truncation=True)
        lock.release()
        labels = labels['input_ids']
        labels = np.where(np.equal(labels, tokenizer.pad_token_id), -100, labels)
        return model_inputs['input_ids'], model_inputs['attention_mask'], labels


def get_dataset(dataset, tokenizer, shuffle=True):
    input_colums=['sentence', 'label', 'text_label']
    output_columns=['input_ids', 'attention_mask', 'labels']
    dataset = dataset.map(MapFunc(input_colums, output_columns),
                          input_colums, output_columns)
    if shuffle:
        dataset = dataset.shuffle(64)
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = get_dataset(train_dataset, tokenizer)
eval_dataset = get_dataset(validation_dataset, tokenizer, shuffle=False)


# In[ ]:


next(train_dataset.create_dict_iterator())


# In[ ]:


from mindhf.core import optim
# optimizer and lr scheduler
optimizer = optim.AdamW(model.trainable_params(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataset) * num_epochs),
)


# In[ ]:


from mindhf.core import value_and_grad
# training and evaluation
def forward_fn(**batch):
    outputs = model(**batch)
    loss = outputs.loss
    return loss

grad_fn = value_and_grad(forward_fn, model.trainable_params())

for epoch in range(num_epochs):
    model.set_train()
    total_loss = 0
    train_total_size = train_dataset.get_dataset_size()
    for step, batch in enumerate(tqdm(train_dataset.create_dict_iterator(), total=train_total_size)):
        optimizer.zero_grad()
        loss = grad_fn(**batch)
        optimizer.step()
        total_loss += loss.float()
        lr_scheduler.step()

    model.set_train(False)
    eval_loss = 0
    eval_preds = []
    eval_total_size = eval_dataset.get_dataset_size()
    for step, batch in enumerate(tqdm(eval_dataset.create_dict_iterator(), total=eval_total_size)):
        with mindspore._no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.float()
        eval_preds.extend(
            tokenizer.batch_decode(ops.argmax(outputs.logits, -1).asnumpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataset)
    eval_ppl = ops.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataset)
    train_ppl = ops.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


# In[ ]:


# print accuracy
correct = 0
total = 0

ground_truth = []

for pred, data in zip(eval_preds, validation_dataset.create_dict_iterator(output_numpy=True)):
    true = str(data['text_label'])
    ground_truth.append(true)
    if pred.strip() == true.strip():
        correct += 1
    total += 1
accuracy = correct / total * 100
print(f"{accuracy=} % on the evaluation dataset")
print(f"{eval_preds[:10]=}")
print(f"{ground_truth[:10]=}")


# In[ ]:


next(eval_dataset.create_tuple_iterator())


# In[ ]:


# saving model
peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)


# In[ ]:


ckpt = f"{peft_model_id}/adapter_model.ckpt"
get_ipython().system('du -h $ckpt')


# In[ ]:


from mindhf.peft import PeftModel, PeftConfig

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)


# In[ ]:


model.set_train(False)
example = next(validation_dataset.create_dict_iterator(output_numpy=True))

print(example['text_label'])
inputs = tokenizer(example['text_label'], return_tensors="ms")
print(inputs)

with mindspore._no_grad():
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(outputs)
    print(tokenizer.batch_decode(outputs.asnumpy(), skip_special_tokens=True))

