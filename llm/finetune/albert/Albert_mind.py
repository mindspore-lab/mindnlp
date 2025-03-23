import random
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import AlbertTokenizer, AlbertForSequenceClassification
from mindnlp.engine import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
import evaluate

# 1. 加载预训练模型和分词器
model_name = "albert-base-v1"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

# 2. 加载IMDb数据集
dataset = load_dataset("stanfordnlp/imdb", trust_remote_code=True)
print("dataset:", dataset)
# 3. 数据预处理函数


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    # 添加标签到返回字典
    tokenized["labels"] = examples["label"]
    return tokenized


# 应用预处理
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 检查标签分布（修正后的代码）
print("\n==== 数据分布验证 ====")

# 检查训练集
train_labels = np.array(tokenized_datasets["train"]["labels"])
print("训练集标签统计:")
print("- 唯一值:", np.unique(train_labels))
print("- 分布:", np.bincount(train_labels))

# 检查测试集
test_labels = np.array(tokenized_datasets["test"]["labels"])
print("\n测试集标签统计:")
print("- 唯一值:", np.unique(test_labels))
print("- 分布:", np.bincount(test_labels))
# 4. 转换数据集格式

def create_dataset(data, batch_size=8):
    # 将数据转换为列表以便打乱
    data_list = list(data)
    random.shuffle(data_list)  # 打乱数据顺序
    
    def generator():
        for item in data_list:  # 遍历打乱后的数据
            yield item["input_ids"], item["attention_mask"], Tensor(item["labels"], dtype=ms.int32)
    
    return GeneratorDataset(generator(), ["input_ids", "attention_mask", "labels"]).batch(batch_size)


train_dataset = create_dataset(tokenized_datasets["train"])
eval_dataset = create_dataset(tokenized_datasets["test"])

# 4. 加载评估指标
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

sample = next(iter(train_dataset))
print("Input IDs:", sample[0])
print("Attention Mask:", sample[1])
print("Labels:", sample[2])

# 自定义指标计算函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred  # 直接解包为logits和labels
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="binary")["f1"],
        "precision": precision.compute(predictions=predictions, references=labels, average="binary")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="binary")["recall"]
    }


# 5. 配置训练参数
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    output_dir="./results",
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",  # 根据准确率选择最佳模型
    greater_is_better=True,            # 准确率越高越好
)

# 6. 初始化并运行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,  # 添加指标计算函数
)

trainer.train()

# 7. 评估模型
eval_results = trainer.evaluate(eval_dataset)
print(f"Evaluation results: {eval_results}")
print("\nFinal evaluation results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")

