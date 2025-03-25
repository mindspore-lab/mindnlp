import mindspore as ms
import mindspore.dataset as ds
from datasets import load_dataset
from mindnlp.transformers import (
    BitForImageClassification,
    AutoImageProcessor
)
from mindnlp.engine import Trainer, TrainingArguments
import os
import numpy as np
ms.set_context(device_target="Ascend")
model_name = "HorcruxNo13/bit-50"
processor = AutoImageProcessor.from_pretrained(model_name)
model = BitForImageClassification.from_pretrained(
    model_name,
    num_labels=102,
    ignore_mismatched_sizes=True
)
dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split="train")
# 将训练集按8:2的比例拆分为训练集和测试集
dataset = dataset.train_test_split(test_size=0.2, seed=42)
dataset.save_to_disk("./flowers102")

print(dataset)
# 选择一个测试集样本进行测试
test_image = dataset['test'][0]['image']
test_label = dataset['test'][0]['label']

print("\n=== 训练参数 ===")
training_args = TrainingArguments(
    output_dir="./mindNLP_bit_flowers102",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    num_train_epochs=10,
    gradient_accumulation_steps=1,
    logging_steps=50,
    load_best_model_at_end=True,
    warmup_steps=0,
    weight_decay=0.01,
    remove_unused_columns=False,
    max_grad_norm=0.0  # 禁用梯度裁剪
)
print("\n=== 先生成np数据 ===")
train_data = []
train_labels = []
for item in dataset['train']:
    img = item['image'].convert('RGB')
    inputs = processor(images=img, return_tensors="np", size={"height": 384, "width": 384})
    train_data.append(inputs['pixel_values'][0])
    train_labels.append(item['label'])
test_data = []
test_labels = []
for item in dataset['test']:
    img = item['image'].convert('RGB')
    inputs = processor(images=img, return_tensors="np", size={"height": 384, "width": 384})
    test_data.append(inputs['pixel_values'][0])
    test_labels.append(item['label'])
train_data = np.array(train_data, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.int32)
test_data = np.array(test_data, dtype=np.float32)
test_labels = np.array(test_labels, dtype=np.int32)
print("\n=== 将预处理后的数据集转换为MindSpore格式 ===")
def create_mindspore_dataset(data, labels, batch_size, shuffle=True):
    dataset = ds.NumpySlicesDataset(
        {
            "pixel_values": data,
            "labels": labels
        },
        shuffle=shuffle
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

# 创建训练和评估数据集
train_dataset = create_mindspore_dataset(
    train_data, 
    train_labels, 
    batch_size=training_args.per_device_train_batch_size,
    shuffle=True
)

eval_dataset = create_mindspore_dataset(
    test_data, 
    test_labels, 
    batch_size=training_args.per_device_eval_batch_size,
    shuffle=False
)

# 单图测试函数
def test_single_image(model, processor, image):
    inputs = processor(
        images=image.convert('RGB'),
        return_tensors="ms",
        size={"height": 384, "width": 384}
    )
    model.set_train(False)
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(-1)
    return predictions.asnumpy().item()

print("\n=== 训练前测试 ===")
pred_before = test_single_image(model, processor, test_image)
print(f"真实标签: {test_label}")
print(f"预测标签: {pred_before}")

import evaluate
import numpy as np
from mindnlp.engine.utils import EvalPrediction

metric = evaluate.load("accuracy")
# 添加调试信息
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    result = metric.compute(predictions=predictions, references=labels)
    return result
print("\n=== 创建Trainer实例 ===")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset
# )
print("\n=== 训练 ===")
trainer.train()
test_results = trainer.evaluate()
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

print("\n=== 训练后测试 ===")
pred_after = test_single_image(model, processor, test_image)
print(f"真实标签: {test_label}")
print(f"预测标签: {pred_after}")
