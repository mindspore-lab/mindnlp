from mindspore.dataset import GeneratorDataset as ds_GeneratorDataset
import numpy as np
from sklearn.metrics import accuracy_score
from mindnlp.engine import TrainingArguments, Trainer
from mindnlp.transformers import AutoImageProcessor, BeitForImageClassification
import mindspore
from mindspore import Tensor
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision.py_transforms import (
    RandomResizedCrop, RandomHorizontalFlip, Resize, CenterCrop, ToTensor, Normalize
)
from datasets import load_dataset

# 加载数据集
train_ds, test_ds = load_dataset(
    'uoft-cs/cifar10', split=['train[:5000]', 'test[:2000]'])
splits = train_ds.train_test_split(test_size=0.1)
train_ds_hf = splits['train']
val_ds_hf = splits['test']
test_ds_hf = test_ds

# 构造标签映射
id2label = {id: label for id, label in enumerate(
    train_ds_hf.features['label'].names)}
label2id = {label: id for id, label in id2label.items()}

# 初始化图像处理器
processor = AutoImageProcessor.from_pretrained(
    'microsoft/beit-base-patch16-224')
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# 定义预处理流程
normalize = Normalize(mean=image_mean, std=image_std)
transform_train = Compose([
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])
transform_val = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

# 定义Hugging Face数据变换


def train_transforms(examples):
    examples['pixel_values'] = [transform_train(
        image.convert("RGB")) for image in examples['img']]
    return examples


def val_transforms(examples):
    examples['pixel_values'] = [transform_val(
        image.convert("RGB")) for image in examples['img']]
    return examples


# 应用transform到原始数据集
train_ds_hf.set_transform(train_transforms)
val_ds_hf.set_transform(val_transforms)
test_ds_hf.set_transform(val_transforms)

# 创建 MindSpore Dataset


def create_mindspore_dataset(hf_dataset):
    def generator():
        for example in hf_dataset:
            # 获取图像数据
            pixel_data = np.array(example['pixel_values'], dtype=np.float32)

            # 中间打印调试
            # print("Raw pixel_data shape:", pixel_data.shape)  #  (C, H, W)  (3, 224, 224)

            # 处理图像数据维度
            if pixel_data.ndim == 4 and pixel_data.shape[0] == 1:
                #  (1, C, H, W)
                pixel_data = pixel_data.squeeze(0)

            yield pixel_data, np.int32(example['label'])

    return ds_GeneratorDataset(
        generator,
        column_names=['pixel_values', 'labels'],
        column_types=[mindspore.float32, mindspore.int32]
    )


#  创建数据集
train_ds = create_mindspore_dataset(train_ds_hf).batch(
    10, drop_remainder=True)  # 10个样本
val_ds = create_mindspore_dataset(val_ds_hf).batch(4, drop_remainder=True)
test_ds = create_mindspore_dataset(test_ds_hf).batch(4, drop_remainder=True)

# 中间打印调试
# for batch in train_ds.create_tuple_iterator():
#    pixel_batch, label_batch = batch
#    print("Batch shape:", pixel_batch.shape)  # 格式 (10, 3, 224, 224)
#    break

# 加载模型
# 初始化训练参数
args = TrainingArguments(
    output_dir="checkpoints",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    remove_unused_columns=False,
    max_grad_norm=0.0,  # 禁用梯度裁剪 否则 Infer type failed.
)

# 初始化模型
model = BeitForImageClassification.from_pretrained(
    'microsoft/beit-base-patch16-224',
    num_labels=10,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,

)

# 定义评估指标


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(predictions, labels)}


# 初始化Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

# 开始训练
trainer.train()
