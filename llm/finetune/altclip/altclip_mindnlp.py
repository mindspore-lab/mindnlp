import os
# 设置Hugging Face镜像源，加速模型下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mindspore as ms
from mindspore import nn, context, Tensor, save_checkpoint
from mindspore.common.initializer import Normal
from mindspore.dataset import Cifar10Dataset, GeneratorDataset
from mindspore.dataset.vision import Resize, ToTensor, Normalize
from mindspore.dataset.transforms import TypeCast
import mindspore.common.dtype as mstype
from mindnlp.transformers import CLIPImageProcessor, AltCLIPVisionModel, AltCLIPVisionConfig
from sklearn.metrics import accuracy_score, f1_score

context.set_context(mode=context.PYNATIVE_MODE,
                   device_target="Ascend")


model_name = "BAAI/AltCLIP"
image_processor = CLIPImageProcessor.from_pretrained(model_name)
vision_model = AltCLIPVisionModel.from_pretrained(model_name)
vision_config = AltCLIPVisionConfig.from_pretrained(model_name)


for param in vision_model.get_parameters():
    param.requires_grad = True

class ImageClassifier(nn.Cell):
    def __init__(self, vision_model, num_classes=10):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = nn.Dense(vision_model.config.hidden_size, num_classes, weight_init=Normal(0.02))

    def construct(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# 实例化分类模型
model = ImageClassifier(vision_model, num_classes=10)

# 图像处理
mean = image_processor.image_mean
std = image_processor.image_std

transform = [
    Resize((224, 224)),
    ToTensor(),
    TypeCast(mstype.float32),
    Normalize(mean=mean, std=std, is_hwc=False)
]

# 数据加载
def load_cifar10_data(batch_size=64):
    type_cast_label = TypeCast(mstype.int32)
    dataset = Cifar10Dataset("./cifar10_data_bin", usage='train', shuffle=False)
    dataset = dataset.map(operations=transform, input_columns="image")

    images, labels = [], []
    for item in dataset.create_dict_iterator():
        images.append(item["image"].asnumpy())
        labels.append(item["label"].asnumpy())

    images = np.array(images)
    labels = np.array(labels)

    train_idx, val_idx = train_test_split(np.arange(len(images)), test_size=0.2, random_state=42)
    train_data = [(images[i], labels[i]) for i in train_idx]
    val_data = [(images[i], labels[i]) for i in val_idx]

    train_loader = GeneratorDataset(train_data, column_names=["image", "label"], shuffle=True)
    train_loader = train_loader.map(operations=type_cast_label, input_columns="label").batch(batch_size)

    val_loader = GeneratorDataset(val_data, column_names=["image", "label"], shuffle=False)
    val_loader = val_loader.map(operations=type_cast_label, input_columns="label").batch(batch_size)

    return train_loader, val_loader

# 自定义学习率调度器
class WarmupDecayLR:
    def __init__(self, base_lr, total_steps, warmup_steps):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.warmup_steps:
            return self.base_lr * self.step_num / self.warmup_steps
        else:
            decay_factor = (self.total_steps - self.step_num) / max(self.total_steps - self.warmup_steps, 1)
            return self.base_lr * decay_factor

# 训练模型函数
def train_model(model, train_loader, val_loader, epochs=10, base_lr=2e-5):
    loss_fn = nn.CrossEntropyLoss()

    # 参数分组
    decay_params = []
    no_decay_params = []
    for param in model.trainable_params():
        pname = param.name.lower()
        if "bias" in pname or "layernorm" in pname:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    group_params = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"order_params": model.trainable_params()}
    ]

    optimizer = nn.AdamWeightDecay(params=group_params, learning_rate=base_lr)

    total_steps = train_loader.get_dataset_size() * epochs
    scheduler = WarmupDecayLR(base_lr, total_steps, warmup_steps=int(0.1 * total_steps))

    net_with_loss = nn.WithLossCell(model, loss_fn)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_loss = 0

        for images, labels in tqdm(train_loader.create_tuple_iterator(), desc="Training"):
            lr = scheduler.step()
            optimizer.learning_rate = Tensor(lr, dtype=ms.float32)

            loss = train_network(images, labels)
            total_loss += loss.asnumpy()

        avg_loss = total_loss / train_loader.get_dataset_size()
        print(f"Train Loss: {avg_loss:.4f}")

        # 验证阶段
        model.set_train(False)
        all_preds, all_labels = [], []
        val_loss_total = 0

        for images, labels in tqdm(val_loader.create_tuple_iterator(), desc="Validation"):
            logits = model(images)
            loss = loss_fn(logits, labels)
            val_loss_total += loss.asnumpy()

            preds = logits.argmax(axis=1).asnumpy()
            all_preds.extend(preds)
            all_labels.extend(labels.asnumpy())

        val_loss_avg = val_loss_total / val_loader.get_dataset_size()
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Validation Loss: {val_loss_avg:.4f}")
        print(f"Validation Accuracy: {acc:.4f}")
        print(f"Validation F1 Score: {f1:.4f}")
        model.set_train(True)

    return model

# 主函数入口
def main():
    print("加载 CIFAR-10 数据...")
    train_loader, val_loader = load_cifar10_data(batch_size=32)

    print("开始训练 AltCLIP 图像分类模型（MindNLP）...")
    trained_model = train_model(model, train_loader, val_loader, epochs=10)

    save_checkpoint(trained_model, "altclip_mindnlp.ckpt")
    print("模型已保存至 altclip_mindnlp.ckpt")

if __name__ == "__main__":
    main()
