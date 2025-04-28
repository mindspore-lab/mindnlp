import os
import numpy as np
import random
import mindspore as ms
from mindspore import nn, ops, Tensor, set_seed
from mindspore.dataset import GeneratorDataset
from mindnlp.transformers import AutoModelForSeq2SeqLM, BartphoTokenizer
from mindnlp.engine import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset

import evaluate

# 加载评估指标
sacrebleu_metric = evaluate.load("sacrebleu")

# 定义模型和数据路径
MODEL_NAME = "vinai/bartpho-syllable"
MAX_LENGTH = 32  # 最大序列长度
output_dir = './saved_model_weights'  # 模型保存路径
os.makedirs(output_dir, exist_ok=True)


# 自定义训练回调函数来打印每个epoch的loss
class LossLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        """在每个epoch结束时调用"""
        # 获取当前训练信息
        epoch = state.epoch
        loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0

        # 打印当前epoch的训练loss
        print(f"Epoch {epoch}: train_loss = {loss:.6f}")

        # 如果有评估结果，也打印出来
        if 'eval_loss' in state.log_history[-1]:
            eval_loss = state.log_history[-1].get('eval_loss', 0.0)
            eval_metric = state.log_history[-1].get('eval_sacrebleu', 0.0)
            print(f"Epoch {epoch}: eval_loss = {eval_loss:.6f}, eval_sacrebleu = {eval_metric:.4f}")


# 数据预处理函数
def preprocess_function(examples):
    # 对输入和目标文本进行tokenize
    return tokenizer(
        examples["error"],
        text_target=examples["original"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )


# 计算评估指标
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # 如果模型返回的是元组，取第一个元素（预测logits）
    if isinstance(preds, tuple):
        preds = preds[0]

    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 处理标签中的pad token
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 简单的后处理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]  # sacrebleu需要列表的列表

    # 计算BLEU分数
    result = sacrebleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {
        "sacrebleu": round(result["score"], 4)
    }


# 为MindSpore创建数据集
def create_mindspore_dataset(data, batch_size=8):
    data_list = list(data)

    def generator():
        for item in data_list:
            yield (
                Tensor(item["input_ids"], dtype=ms.int32),
                Tensor(item["attention_mask"], dtype=ms.int32),
                Tensor(item["labels"], dtype=ms.int32)
            )

    return GeneratorDataset(
        generator,
        column_names=["input_ids", "attention_mask", "labels"]
    ).batch(batch_size)


# 对logits进行预处理，防止内存溢出
def preprocess_logits_for_metrics(logits, labels):
    """防止内存溢出"""
    pred_ids = ms.mint.argmax(logits[0], dim=-1)
    return pred_ids, labels


# 主函数
def main():
    global tokenizer  # 使tokenizer在函数外可用

    # 加载模型和tokenizer
    print("正在加载模型和tokenizer...")
    tokenizer = BartphoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 加载数据集
    print("正在加载数据集...")
    train_path = './train.csv'
    test_path = './test.csv'
    dataset = load_dataset("csv", data_files={"train": train_path, "test": test_path})

    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"测试集样本数: {len(dataset['test'])}")

    # 数据预处理
    print("正在进行数据预处理...")
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 创建MindSpore数据集
    print("正在创建MindSpore数据集...")
    train_dataset = create_mindspore_dataset(tokenized_datasets["train"], batch_size=8)
    eval_dataset = create_mindspore_dataset(tokenized_datasets["test"], batch_size=8)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=2,
    )

    # 初始化训练器
    print("初始化训练器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[LossLoggerCallback()]
    )

    # 开始训练
    print("开始训练...")
    trainer.train()
    # 保存模型
    print(f"训练完成，保存模型到 {output_dir}...")
    model.save_pretrained(output_dir)
    # 模型评估
    print("进行模型最终评估...")
    eval_results = trainer.evaluate()
    print(f"最终评估结果: {eval_results}")


if __name__ == "__main__":
    main()
