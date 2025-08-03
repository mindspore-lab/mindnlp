import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.optim import AdamW  # 从torch.optim导入AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载BARThez模型和分词器
model_name = "moussaKam/barthez"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)


# 数据集加载和处理
class AllocineDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 加载并处理Allocine数据集
def load_allocine_dataset(sample_ratio=0.5):
    """
    使用Hugging Face的datasets库加载Allocine数据集。
    数据集包含"review"文本和"label"标签（0表示负面，1表示正面）

    参数:
        sample_ratio: 要使用的数据比例，范围(0,1]
    """
    # 加载Allocine数据集
    dataset = load_dataset("allocine")

    # 取数据集的子集(10%)
    if sample_ratio < 1.0:
        train_subset = dataset["train"].shuffle(seed=42).select(range(int(len(dataset["train"]) * sample_ratio)))
        test_subset = dataset["test"].shuffle(seed=42).select(range(int(len(dataset["test"]) * sample_ratio)))

        dataset = {
            "train": train_subset,
            "test": test_subset
        }

    return dataset


# 定义训练函数
def train_model(model, train_dataloader, val_dataloader, epochs=3):
    # 优化器设置
    optimizer = AdamW(model.parameters(), lr=2e-6, eps=1e-8)

    # 学习率调度器
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 记录训练过程
    training_stats = []

    # 开始训练循环
    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch + 1} / {epochs} ========")

        # 训练
        model.train()
        total_train_loss = 0

        train_progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
        for batch in train_progress_bar:
            # 清除之前计算的梯度
            optimizer.zero_grad()

            # 将数据移动到GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()
            scheduler.step()

        # 计算平均训练损失
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # 评估
        model.eval()
        total_eval_loss = 0
        predictions = []
        true_labels = []

        for batch in tqdm(val_dataloader, desc="Validation", leave=True):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_eval_loss += loss.item()

                # 获取预测结果
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        # 计算评估指标
        avg_val_loss = total_eval_loss / len(val_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"Validation loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # 保存训练统计信息
        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Accuracy': accuracy,
            'F1 Score': f1
        })

    return training_stats


# 主程序
def main():
    # 加载数据，只使用50%的数据
    print("正在加载Allocine数据集(50%)...")
    dataset = load_allocine_dataset(sample_ratio=0.5)

    # Allocine数据集已经分割为训练集和测试集
    train_dataset_raw = dataset["train"]
    test_dataset_raw = dataset["test"]

    print(f"训练样本数: {len(train_dataset_raw)}")
    print(f"测试样本数: {len(test_dataset_raw)}")

    # 创建自定义数据集实例
    train_dataset = AllocineDataset(
        train_dataset_raw["review"],
        train_dataset_raw["label"],
        tokenizer
    )

    val_dataset = AllocineDataset(
        test_dataset_raw["review"],
        test_dataset_raw["label"],
        tokenizer
    )

    # 创建数据加载器
    batch_size = 16  # 减小批量大小以减少内存使用
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # 训练模型
    print("开始训练...")
    training_stats = train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=10
    )

    # 保存模型
    output_dir = './barthez_allocine_model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n训练完成!")



if __name__ == "__main__":
    main()