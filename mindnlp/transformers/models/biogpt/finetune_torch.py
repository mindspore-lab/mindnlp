import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptForCausalLM, BioGptTokenizer, get_linear_schedule_with_warmup
from peft import PrefixTuningConfig, get_peft_model, TaskType
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
import re

# 设置超参数
max_length = 512
lr = 1e-5
num_epochs = 20
batch_size = 16

# 设置输出目录
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"outputs/{current_time}"

# 加载 BioGPT 模型和 Tokenizer
peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=9)
model_path = '/tmp/pretrainmodel/biogpt'
model = BioGptForCausalLM.from_pretrained(model_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
tokenizer = BioGptTokenizer.from_pretrained(model_path)

# 数据加载和预处理
def process_func(question, context, label, tokenizer, max_len=max_length):
    context = ' '.join(sen.strip() for sen in context)
    context = re.sub(r'\n', ' ', context)
    context = re.sub(r'\s+', ' ', context)
    texts = "question: {} context: {} ".format(question.strip(), context.strip())
    labels = 'the answer to the question given the context is ' + label.strip() + '.'
    
    # 使用 tokenizer 对文本进行编码
    text_encoding = tokenizer(texts, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
    labels_encoding = tokenizer(labels, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
    
    input_ids = text_encoding["input_ids"].flatten()
    attention_mask = text_encoding["attention_mask"].flatten()
    labels = labels_encoding["input_ids"].flatten()
    
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# 加载数据集
dataset_path = 'PubMedQA'
with open(os.path.join(dataset_path, 'train_set.json'), 'r') as file:
    train_set = json.load(file)
with open(os.path.join(dataset_path, 'dev_set.json'), 'r') as file:
    val_set = json.load(file)
with open(os.path.join(dataset_path, 'test_set.json'), 'r') as file:
    test_set = json.load(file)

# 处理训练、验证和测试数据
processed_train_data = [process_func(item['QUESTION'], item['CONTEXTS'], item['final_decision'], tokenizer) for item in train_set.values()]
processed_val_data = [process_func(item['QUESTION'], item['CONTEXTS'], item['final_decision'], tokenizer) for item in val_set.values()]
processed_test_data = [process_func(item['QUESTION'], item['CONTEXTS'], item['final_decision'], tokenizer) for item in test_set.values()]

# 创建 PyTorch Dataset
class PubMedQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 创建 DataLoader
train_dataset = PubMedQADataset(processed_train_data)
eval_dataset = PubMedQADataset(processed_val_data)
test_dataset = PubMedQADataset(processed_test_data)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 定义优化器和学习率调度器
optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(train_dataloader) * num_epochs
# lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练和评估
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}")

    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
    
    print(f"Epoch {epoch}, Eval Loss: {eval_loss / len(eval_dataloader)}, learning_rage: {optimizer.param_groups[0]['lr']}")

# 保存模型和 Tokenizer
model.save_pretrained(os.path.join(output_dir, "biogptmodel-finetuned"))
