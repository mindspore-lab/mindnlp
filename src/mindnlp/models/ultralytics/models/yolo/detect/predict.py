import os
import numpy as np
import cv2
import mindspore as ms

from engine.predictor import BasePredictor
from utils.ops import non_max_suppression  

# 结果封装与可视化类
class Results:
    """
    目标检测结果封装类
    提供统一的内部接口以供调用方提取检测框、分类结果及可视化
    """
    def __init__(self, orig_img, det, names):
        self.orig_img = orig_img
        self.det = det  # [N, 6] 格式：x1, y1, x2, y2, conf, cls
        self.names = names

    def save(self, save_dir=".", file_name="result.jpg"):
        """
        在原始图像上绘制边界框及置信度标签并持久化保存
        """
        res_img = self.orig_img.copy()
        
        for *xyxy, conf, cls in self.det:
            bx1, by1, bx2, by2 = map(int, xyxy)
            cls_id = int(cls)
            label_text = self.names.get(cls_id, f"ID:{cls_id}")
            
            # 绘制绿色边界框
            color = (0, 255, 0)
            cv2.rectangle(res_img, (bx1, by1), (bx2, by2), color, 2)
            
            # 绘制带背景底色的文本标签，提升可视化对比度
            text = f"{label_text} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(res_img, (bx1, by1 - 20), (bx1 + tw, by1), color, -1)
            cv2.putText(res_img, text, (bx1, by1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, res_img)

# 检测任务专用推理算子
class DetectionPredictor(BasePredictor):
    """
    YOLO11 目标检测专属推理调度器
    基于 BasePredictor 的预处理流水线
    """
    def setup_model(self, model, ckpt_path=None):
        """挂载模型结构并提取类别字典映射"""
        super().setup_model(model, ckpt_path)
        
        if hasattr(self.model, 'names'):
            self.names = self.model.names
        else:
            nc = getattr(self.model.config, 'nc', 80) if hasattr(self.model, 'config') else 80
            self.names = {i: f"class_{i}" for i in range(nc)}

    def postprocess(self, preds, orig_img, prep_info):
        """
        检测任务后处理核心逻辑：原始输出解析 -> NMS 算子 -> 坐标映射还原 -> 结果实体封装
        """
        # 兼容性处理
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # 执行非极大值抑制，依据超参数配置筛除低置信度及高度重叠框
        preds_nms = non_max_suppression(
            preds, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres
        )
        
        # 提取单一批次的检测结果，形状转化为 NumPy Array 以加速标量计算
        det = preds_nms[0].asnumpy() if len(preds_nms) > 0 else np.zeros((0, 6))

        # 逆向尺度还原：将在网络输入尺度下的边界框映射回原始图像尺度
        ratio, pad_w, pad_h = prep_info
        if len(det) > 0:
            det[:, 0] = (det[:, 0] - pad_w) / ratio
            det[:, 1] = (det[:, 1] - pad_h) / ratio
            det[:, 2] = (det[:, 2] - pad_w) / ratio
            det[:, 3] = (det[:, 3] - pad_h) / ratio

            h0, w0 = orig_img.shape[:2]
            det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, w0)
            det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, h0)

        return Results(orig_img, det, self.names)