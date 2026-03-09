import os
import numpy as np
import cv2
import mindspore as ms

from engine.predictor import BasePredictor
from utils.ops import non_max_suppression  

# 实例分割结果实体与可视化类
class Results:
    """
    分割结果封装类
    提供在原图上叠印目标边界框及像素级掩码 (Mask) 的持久化保存方法
    """
    def __init__(self, orig_img, det, masks, names):
        self.orig_img = orig_img
        self.det = det      # [N, 6] 格式：x1, y1, x2, y2, conf, cls
        self.masks = masks  # 包含 N 个掩码矩阵的列表，尺寸与 orig_img 一致
        self.names = names

    def save(self, save_dir=".", file_name="result.jpg"):
        """将预测边界框与掩码着色后保存为图像文件"""
        res_img = self.orig_img.copy()
        
        for i, (*xyxy, conf, cls) in enumerate(self.det):
            bx1, by1, bx2, by2 = map(int, xyxy)
            cls_id = int(cls)
            label_text = self.names.get(cls_id, f"ID:{cls_id}")
            
            # 掩码渲染 (橙色通透叠加处理)
            if i < len(self.masks):
                mask_orig = self.masks[i]
                color = np.array([255, 120, 0], dtype=np.uint8)  # 设定掩码基础色 (BGR)
                weight_mask = (np.clip(mask_orig, 0, 1) ** 2.0)[:, :, None]
                overlay = (weight_mask * color).astype(np.uint8)
                res_img = cv2.addWeighted(res_img, 1.0, overlay, 0.45, 0)

                # 计算并绘制掩码轮廓的平滑描边
                binary_mask = (mask_orig > 0.4).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                cv2.drawContours(res_img, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

            # 边界框与文本标签渲染
            cv2.rectangle(res_img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            text = f"{label_text} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(res_img, (bx1, by1 - 20), (bx1 + tw, by1), (0, 255, 0), -1)
            cv2.putText(res_img, text, (bx1, by1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, res_img)


# 实例分割推理算子
class SegmentationPredictor(BasePredictor):
    """
    YOLO11 实例分割任务预测器
    扩展基类能力以解析掩码系数，并在特征映射后执行 Mask 合成
    """
    def setup_model(self, model, ckpt_path=None):
        super().setup_model(model, ckpt_path)
        if hasattr(self.model, 'names'):
            self.names = self.model.names
        else:
            nc = getattr(self.model.config, 'nc', 80) if hasattr(self.model, 'config') else 80
            self.names = {i: f"class_{i}" for i in range(nc)}

    def postprocess(self, preds, orig_img, prep_info):
        """
        后处理解析：分离预测边界框与掩码系数 -> NMS -> 合成实例掩码 -> 坐标空间映射
        """
        # 拆解网络多头输出：(预测矩阵, 原型掩码特征图)
        prediction, proto = preds
        
        # 应用非极大值抑制。内部会提取附带的 32 维 Mask 系数
        preds_nms = non_max_suppression(
            prediction, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres
        )
        
        det = preds_nms[0].asnumpy() if len(preds_nms) > 0 else np.zeros((0, 38))
        if len(det) == 0:
            return Results(orig_img, np.zeros((0, 6)), [], self.names)

        # 剥离空间边界框数据与对应的特征系数
        boxes_640 = det[:, :4].copy()
        mask_coeffs = det[:, 6:]

        # 在网络输入尺度 (通常为 640x640) 下合成掩码实例
        proto_data = proto.asnumpy()[0]
        masks_640 = self._process_mask_ultimate_precision(
            proto_data, mask_coeffs, boxes_640, (self.imgsz, self.imgsz)
        )

        # 执行边界框的逆向尺度还原
        ratio, pad_w, pad_h = prep_info
        det[:, 0] = (det[:, 0] - pad_w) / ratio
        det[:, 1] = (det[:, 1] - pad_h) / ratio
        det[:, 2] = (det[:, 2] - pad_w) / ratio
        det[:, 3] = (det[:, 3] - pad_h) / ratio

        h0, w0 = orig_img.shape[:2]
        det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, w0)
        det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, h0)

        # 掩码阵列反向映射：剥离 Padding 后缩放至原始图像空间
        final_masks_orig = []
        top, bottom = int(round(pad_h - 0.1)), int(round(self.imgsz - pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(self.imgsz - pad_w + 0.1))
        
        for m_img in masks_640:
            m_cropped = m_img[top:bottom, left:right]
            mask_orig = cv2.resize(m_cropped, (w0, h0), interpolation=cv2.INTER_LINEAR)
            final_masks_orig.append(mask_orig)

        return Results(orig_img, det[:, :6], final_masks_orig, self.names)

    def _process_mask_ultimate_precision(self, protos, masks_in, bboxes, shape):
        """
        利用矩阵乘法结合 Sigmoid 激活还原特征图概率，并应用边界框进行严格裁剪
        """
        c, mh, mw = protos.shape  
        ih, iw = shape  

        masks = (masks_in @ protos.reshape(c, -1)).reshape(-1, mh, mw)
        width_ratio, height_ratio = mw / iw, mh / ih
        
        scaled_bboxes = bboxes.copy()
        scaled_bboxes[:, [0, 2]] *= width_ratio
        scaled_bboxes[:, [1, 3]] *= height_ratio

        full_masks = []
        for i in range(len(masks)):
            m_raw = masks[i]
            m_min, m_max = m_raw.min(), m_raw.max()
            m_norm = (m_raw - m_min) / (m_max - m_min + 1e-6)
            m_prob = 1 / (1 + np.exp(-((m_norm - 0.5) * 10)))

            x1, y1, x2, y2 = scaled_bboxes[i]
            r = np.arange(mw, dtype=np.float32)[None, :]
            col = np.arange(mh, dtype=np.float32)[:, None]
            
            # 使用逻辑张量截断框外冗余特征
            m_prob = m_prob * ((r >= x1) * (r < x2) * (col >= y1) * (col < y2))
            
            m_img = cv2.resize(m_prob, (iw, ih), interpolation=cv2.INTER_CUBIC)
            full_masks.append(m_img)

        return full_masks