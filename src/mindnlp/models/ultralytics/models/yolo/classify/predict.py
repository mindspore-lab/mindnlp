import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import mindspore as ms
from mindspore import Tensor, ops

from engine.predictor import BasePredictor

# 分类结果实体类
class Results:
    """图像分类结果的标准化封装实体，提供置信度解析与可视化保存接口"""
    
    def __init__(self, orig_img, probs, names=None):
        self.orig_img = orig_img
        self.probs = probs
        self.names = names or {i: f"class_{i}" for i in range(len(probs))}

    @property
    def top1(self):
        """获取最高置信度类别的索引与概率值"""
        idx = int(np.argmax(self.probs))
        return idx, float(self.probs[idx])

    @property
    def top1_name(self):
        """获取最高置信度类别的映射名称"""
        return self.names.get(self.top1[0], str(self.top1[0]))

    def save(self, save_dir=".", file_name="result.jpg"):
        """将分类结果绘制于原图并保存"""
        # 类型安全防御：确保图像格式兼容 PIL
        if isinstance(self.orig_img, np.ndarray):
            img_drawn = Image.fromarray(cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB))
        else:
            img_drawn = self.orig_img.copy()
            
        draw = ImageDraw.Draw(img_drawn)
        
        idx, conf = self.top1
        text = f"Pred: {self.top1_name}\nConf: {conf:.4f}"
        
        # 绘制背景标签底板与文字
        draw.rectangle([0, 0, 150, 40], fill="black")
        draw.text((5, 5), text, fill="white")
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        img_drawn.save(save_path)


# 分类任务预测器
class ClassificationPredictor(BasePredictor):
    """图像分类任务专用预测器"""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.imgsz = getattr(self.cfg, 'imgsz', 224)
        self.names = {}

    def setup_model(self, model, ckpt_path=None):
        """加载模型权重，并提取类别映射字典以供可视化使用"""
        super().setup_model(model, ckpt_path)
        
        if hasattr(self.model, 'names'):
            self.names = self.model.names
        else:
            nc = getattr(self.model.config, 'nc', 1000) if hasattr(self.model, 'config') else 1000
            self.names = {i: f"class_{i}" for i in range(nc)}

    def preprocess(self, img_source):
        """
        预处理流水线：应用中心裁剪与 ImageNet 标准化，替代基础的 LetterBox 缩放
        """
        if isinstance(img_source, str):
            img = Image.open(img_source).convert('RGB')
        else:
            # 兼容 ndarray 类型的输入
            img = Image.fromarray(cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB))
            
        orig_img = img.copy()

        # 1. 保持宽高比，缩放短边至 256
        w, h = img.size
        short_side = 256
        if w < h:
            new_w, new_h = short_side, int(short_side * h / w)
        else:
            new_w, new_h = int(short_side * w / h), short_side
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # 2. 中心裁剪至目标输入尺寸 (默认 224)
        crop_size = self.imgsz
        left = (new_w - crop_size) / 2
        top = (new_h - crop_size) / 2
        img_cropped = img_resized.crop((left, top, left + crop_size, top + crop_size))

        # 3. 像素值归一化与 ImageNet 数据集统计分布标准化
        img_data = np.array(img_cropped).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_data = (img_data - mean) / std
        
        # 4. 调整通道维度并扩增 Batch 轴
        img_data = img_data.transpose(2, 0, 1)
        im_tensor = Tensor(img_data, ms.float32).expand_dims(0)
        
        return im_tensor, orig_img, None

    def postprocess(self, preds, orig_img, preprocess_info):
        """
        解析分类模型输出特征：提取 Softmax 概率分布并封装结果实体
        """
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
            
        # 提取 Batch 首位的分类 logits 并转化为概率空间
        probs = ops.softmax(preds, axis=1).asnumpy()[0]
        
        res = Results(orig_img=orig_img, probs=probs, names=self.names)
        return res