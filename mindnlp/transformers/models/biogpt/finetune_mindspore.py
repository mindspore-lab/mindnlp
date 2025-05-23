import mindspore
from mindnlp.transformers import BioGptTokenizer, BioGptForCausalLM
from mindnlp.peft import get_peft_model, PrefixTuningConfig, TaskType
import mindspore.dataset as ds
from mindnlp.engine.train_args import TrainingArguments
from mindnlp.engine import Trainer
import numpy as np
import os
from datetime import datetime
from mindspore import context
import re
import json

max_length = 512
lr = 1e-5
num_epochs = 20
warm_epochs = 0
batch_size = 16

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"output/{current_time}"

context.set_context(device_target="Ascend")
peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=9)
# 加载BioGPT模型和tokenizer

model_name = 'microsoft/biogpt'
local_model_path = ".mindnlp/model/microsoft/biogpt"
model = BioGptForCausalLM.from_pretrained(local_model_path)
tokenizer = BioGptTokenizer.from_pretrained(local_model_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 数据加载和预处理
def process_func(question, context, long_answer, label, tokenizer, max_len=max_length):
    context = ' '.join(sen.strip() for sen in context)
    context = re.sub(r'\n', ' ', context)
    # remove duplicate spaces
    context = re.sub(r'\s+', ' ', context)
    texts = "question: {}.context: {}.answer:{}.".format(question.strip(), context.strip(), long_answer.strip())
    labels = 'the answer to the question given the context is ' + label.strip() + '.'
    
    # 使用 tokenizer 对文本进行编码
    text_encoding = tokenizer(texts)
    labels_encoding = tokenizer(labels)
    input_ids = text_encoding["input_ids"] + labels_encoding["input_ids"]
    attention_mask = text_encoding["attention_mask"] + labels_encoding["attention_mask"]
    labels = [-100] * len(text_encoding["input_ids"]) + labels_encoding["input_ids"]
    if len(input_ids) >  max_len:  # 做一个截断
        input_ids = input_ids[: max_len]
        attention_mask = attention_mask[: max_len]
        labels = labels[: max_len]
    else:
        pad_lenth = max_len - len(input_ids)
        input_ids +=[-100]*pad_lenth
        attention_mask +=[0]*pad_lenth
        labels += [-100]*pad_lenth

    return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}

# 加载数据集
dataset_path = 'PubMedQA'
train_set = json.load(open(os.path.join(dataset_path, 'train_set.json'), 'r'))
val_set = json.load(open(os.path.join(dataset_path, 'dev_set.json'), 'r'))
test_set = json.load(open(os.path.join(dataset_path, 'test_set.json'), 'r'))

processed_train_data = [process_func(item['QUESTION'], item['CONTEXTS'], item['LONG_ANSWER'], item['final_decision'], tokenizer) for item in train_set.values()]
processed_val_data = [process_func(item['QUESTION'], item['CONTEXTS'], item['LONG_ANSWER'], item['final_decision'], tokenizer) for item in val_set.values()]
processed_test_data = [process_func(item['QUESTION'], item['CONTEXTS'], item['LONG_ANSWER'], item['final_decision'], tokenizer) for item in test_set.values()]
    
print(len(processed_train_data), len(processed_val_data), len(processed_test_data))

# 创建 Dataset
def my_generator(dataset):
    for item in dataset:
        yield (
            np.array(item["input_ids"], dtype=np.int32),
            np.array(item["attention_mask"], dtype=np.float32),
            np.array(item["labels"], dtype=np.int32)
        )
        
# 创建 GeneratorDataset
print('create dataset')
train_dataset = ds.GeneratorDataset(lambda: my_generator(processed_train_data), column_names=["input_ids", "attention_mask", "labels"]).batch(batch_size)
eval_dataset = ds.GeneratorDataset(lambda: my_generator(processed_val_data), column_names=["input_ids", "attention_mask", "labels"]).batch(batch_size)
test_dataset = ds.GeneratorDataset(lambda: my_generator(processed_test_data), column_names=["input_ids", "attention_mask", "labels"]).batch(batch_size)

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_epochs,  # 根据需要调整训练轮数
    per_device_train_batch_size=batch_size,  # 根据硬件资源调整
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    warmup_steps=warm_epochs,
    weight_decay=0,
    save_total_limit=2,
    save_steps=10,
    evaluation_strategy="epoch",
    logging_dir=output_dir,  # 日志保存目录
    logging_steps=1,
    lr_scheduler_type="constant"
)
print(f"Logging directory: {training_args.logging_dir}")
# 定义 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始训练
print("training start!")
trainer.train()

# 保存模型和 Tokenizer
model.save_pretrained(os.path.join(output_dir, "biogptmodel-finetuned"))
