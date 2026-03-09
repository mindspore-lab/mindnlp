import os
import numpy as np
import cv2
import mindspore as ms
from engine.predictor import BasePredictor
from utils.ops import non_max_suppression

# 预测结果封装与渲染类
class Results:
    """
    姿态估计结果封装类
    负责存储原始图像、边界框检测结果、关键点坐标，并提供可视化渲染与保存功能
    """
    def __init__(self, orig_img, det, kpts, names):
        self.orig_img = orig_img
        self.det = det    # 边界框与类别信息: [N, 6] -> x1, y1, x2, y2, conf, cls
        self.kpts = kpts  # 关键点信息: [N, 17, 3] -> x, y, conf
        self.names = names
        
        # COCO 数据集标准的 17 关键点物理连接拓扑结构 (骨架)
        self.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], 
            [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], 
            [1, 3], [2, 4], [3, 5], [4, 6]
        ]

    def save(self, save_dir=".", file_name="result.jpg"):
        """将边界框、有效关键点及其骨架连线渲染至原图并保存"""
        res_img = self.orig_img.copy()
        
        for i, (*xyxy, conf, cls) in enumerate(self.det):
            # 1. 渲染检测边界框与置信度
            bx1, by1, bx2, by2 = map(int, xyxy)
            cv2.rectangle(res_img, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(res_img, f"{self.names.get(int(cls), 'Pose')} {conf:.2f}", 
                        (bx1, max(by1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 2. 渲染关键点与骨架拓扑
            if i < len(self.kpts):
                kpts = self.kpts[i]
                
                # 绘制有效关键点 (置信度阈值 > 0.5)
                for kx, ky, kconf in kpts:
                    if kconf > 0.5:
                        cv2.circle(res_img, (int(kx), int(ky)), 5, (0, 255, 0), -1)
                
                # 绘制骨架连线 (仅当两端关键点均有效时连线)
                for sk in self.skeleton:
                    p1, p2 = kpts[sk[0]], kpts[sk[1]]
                    if p1[2] > 0.5 and p2[2] > 0.5:
                        cv2.line(res_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 
                                 (255, 255, 0), 2, cv2.LINE_AA)

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        cv2.imwrite(save_path, res_img)
        print(f"[Info] 渲染结果已保存至: {save_path}")

# 姿态预测器核心逻辑
class PosePredictor(BasePredictor):
    """
    YOLO11 姿态估计预测器子类
    继承自 BasePredictor，扩展了关键点维度的后处理与坐标系逆映射逻辑
    """
    def setup_model(self, model, ckpt_path=None):
        """初始化模型参数及拓扑配置"""
        super().setup_model(model, ckpt_path)
        self.nkpt = getattr(self.model.config, 'kpt_shape', [17, 3])[0]
        self.names = getattr(self.model, 'names', {0: 'person'})

    def postprocess(self, preds, orig_img, prep_info):
        """
        推理后处理：执行 NMS，并将坐标逆映射回原始图像尺度
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        # 1. 执行非极大值抑制 (NMS)
        preds_nms = non_max_suppression(
            preds, 
            conf_thres=self.conf_thres, 
            iou_thres=self.iou_thres, 
            nc=len(self.names)
        )
        
        det = preds_nms[0].asnumpy() if len(preds_nms) > 0 else np.zeros((0, 6 + self.nkpt * 3))
        if len(det) == 0:
            return Results(orig_img, np.zeros((0, 6)), [], self.names)

        # 2. 获取预处理时的缩放比例与填充信息
        ratio, pad_w, pad_h = prep_info

        # 3. 逆仿射变换：还原边界框坐标
        det[:, :4] = (det[:, :4] - [pad_w, pad_h, pad_w, pad_h]) / ratio

        # 4. 逆仿射变换：还原关键点坐标并重塑维度
        kpts = det[:, 6:].reshape(-1, self.nkpt, 3)
        kpts[..., 0] = (kpts[..., 0] - pad_w) / ratio
        kpts[..., 1] = (kpts[..., 1] - pad_h) / ratio

        # 5. 越界裁剪：确保所有坐标不超出原始图像边界
        h0, w0 = orig_img.shape[:2]
        det[:, [0, 2]] = np.clip(det[:, [0, 2]], 0, w0)
        det[:, [1, 3]] = np.clip(det[:, [1, 3]], 0, h0)
        kpts[..., 0] = np.clip(kpts[..., 0], 0, w0)
        kpts[..., 1] = np.clip(kpts[..., 1], 0, h0)

        return Results(orig_img, det[:, :6], kpts, self.names)