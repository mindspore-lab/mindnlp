import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import (
    CLIPImageProcessor,
    AltCLIPVisionModel,
    AltCLIPVisionConfig,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载预训练模型与图像处理器
model_name = "BAAI/AltCLIP"
image_processor = CLIPImageProcessor.from_pretrained(model_name)
vision_model = AltCLIPVisionModel.from_pretrained(model_name)
vision_config = AltCLIPVisionConfig.from_pretrained(model_name)
vision_model.to(device)

# 解冻全部参数
for param in vision_model.parameters():
    param.requires_grad = True

# 定义带分类头的模型
class ImageClassifier(torch.nn.Module):
    def __init__(self, vision_model, num_classes=10):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = torch.nn.Linear(vision_config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

model = ImageClassifier(vision_model, num_classes=10).to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# 加载数据集（仅 train 和 val）
def load_cifar10_data(root='./data', batch_size=64):
    full_train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    indices = list(range(len(full_train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# 模型训练 + 验证函数
def train_model(model, train_loader, val_loader, epochs=10):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

    return model

# 主函数
def main():
    print("加载 CIFAR-10 数据集...")
    train_loader, val_loader = load_cifar10_data(batch_size=32)

    print("开始训练 AltCLIP 图像分类模型...")
    trained_model = train_model(model, train_loader, val_loader, epochs=10)

    output_dir = "./cifar10_altclip_model"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(output_dir, "altclip_cifar10.pth"))
    print(f"模型已保存至 {output_dir}")

if __name__ == "__main__":
    main()
