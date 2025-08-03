import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import pandas as pd
import logging
import random
import warnings
import itertools

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 设置种子和配置参数
MODEL_NAME = "vinai/bartpho-syllable"
MAX_LENGTH = 32  # 保持较小的最大长度以获得更好的性能
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 5
WARMUP_STEPS = 500
OUTPUT_DIR = './custom_model_weights'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置HuggingFace镜像（如果需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载数据集
logger.info("正在加载数据集...")
train_path = './train.csv'
test_path = './test.csv'
dataset = load_dataset("csv", data_files={"train": train_path, "test": test_path})

# 加载tokenizer
logger.info(f"正在加载tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
logger.info(f"已加载Tokenizer: {MODEL_NAME}")

# 加载评估指标
metric = evaluate.load("/root/autodl-tmp/evaluate/metrics/sacrebleu")
logger.info(f"已加载评估指标")


# 自定义数据集类
class VietnameseTextCorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        error_text = item['error']
        original_text = item['original']

        # 对输入文本进行编码
        inputs = self.tokenizer(
            error_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 对目标文本进行编码
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                original_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()

        # 将padding的token替换为-100，这样在计算损失时会被忽略
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


# 创建数据集实例
logger.info("创建数据集实例...")
train_dataset = VietnameseTextCorrectionDataset(dataset['train'], tokenizer, MAX_LENGTH)
val_dataset = VietnameseTextCorrectionDataset(dataset['test'], tokenizer, MAX_LENGTH)

logger.info(f"训练集大小: {len(train_dataset)}")
logger.info(f"验证集大小: {len(val_dataset)}")

# 创建数据加载器
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# 加载模型
logger.info(f"正在加载模型: {MODEL_NAME}")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 将模型移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
model.to(device)

# 设置优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 计算总训练步数并设置学习率调度器（不使用梯度累积）
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)


# 使用evaluate库的计算BLEU分数的函数
def compute_bleu(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 处理标签，替换-100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 一些简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    # 使用evaluate加载的sacrebleu计算指标
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    # 打印一些示例进行调试
    # for i in range(min(3, len(decoded_preds))):
    #     logger.debug(f"预测: {decoded_preds[i]}")
    #     logger.debug(f"参考: {decoded_labels[i][0]}")
    #     logger.debug("---")

    return result["score"]


# 评估函数
def evaluate(full_validation=False):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    # 在训练期间进行快速验证，使用验证数据的子集
    eval_dataloader = val_dataloader if full_validation else itertools.islice(val_dataloader, 10)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 评估损失
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            val_loss += loss.item()

            # 使用优化的参数生成预测
            generation_params = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'max_length': MAX_LENGTH,
            }

            if full_validation:
                generation_params['num_beams'] = 4  # 在完整验证中使用束搜索

            generated_ids = model.generate(**generation_params)

            all_preds.extend(generated_ids.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    # 计算平均验证损失
    divisor = len(val_dataloader) if full_validation else 10
    avg_val_loss = val_loss / divisor

    # 计算BLEU分数
    bleu_score = compute_bleu(all_preds, np.array(all_labels))

    return bleu_score, avg_val_loss


# 训练函数（不使用梯度累积）
def train():
    logger.info("开始训练...")

    # 记录最佳模型的BLEU分数
    best_bleu = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in progress_bar:
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # 直接反向传播
            loss.backward()

            # 立即更新参数
            optimizer.step()

            # 更新学习率
            scheduler.step()

            # 累计损失
            total_loss += loss.item()

            # 更新进度条
            progress_bar.set_description(f"轮次 {epoch + 1}/{EPOCHS} | 损失: {total_loss / (step + 1):.4f}")

            # 保存中间检查点
            if step > 0 and step % 1000 == 0:
                logger.info(f"轮次 {epoch + 1}, 步骤 {step}: 损失 = {total_loss / (step + 1):.4f}")

        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"轮次 {epoch + 1}/{EPOCHS} - 平均训练损失: {avg_train_loss:.4f}")

        # 在训练期间进行快速验证
        bleu_score, val_loss = evaluate(full_validation=False)
        logger.info(f"轮次 {epoch + 1}/{EPOCHS} - 快速验证 - 损失: {val_loss:.4f}, BLEU: {bleu_score:.4f}")

        # 只在特定间隔进行完整验证
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            logger.info("执行完整验证...")
            full_bleu_score, full_val_loss = evaluate(full_validation=True)
            logger.info(
                f"轮次 {epoch + 1}/{EPOCHS} - 完整验证 - 损失: {full_val_loss:.4f}, BLEU: {full_bleu_score:.4f}")
            bleu_score = full_bleu_score

        # 保存最佳模型
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            save_path = os.path.join(OUTPUT_DIR, f"best_model_epoch_{epoch + 1}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"在轮次 {epoch + 1} 保存了新的最佳模型，BLEU分数为: {best_bleu:.4f}")

    # 训练完成后保存最终模型
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"训练完成！最终模型保存在 {final_path}")

    return best_bleu


if __name__ == "__main__":
    # 开始训练
    best_bleu = train()
    logger.info(f"训练完成，最佳BLEU分数为: {best_bleu:.4f}")
