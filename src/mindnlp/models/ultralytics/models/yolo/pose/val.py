import numpy as np
import mindspore as ms
from mindspore import ops

from engine.validator import BaseValidator
from utils.ops import non_max_suppression, xywh2xyxy_np
from utils.metrics import PoseMetrics, kpt_iou

class PoseValidator(BaseValidator):
    """
    YOLO11 姿态估计专属验证器
    继承自 BaseValidator，处理数据加载、前向推理、NMS后处理以及 OKS 指标计算逻辑
    """
    def __init__(self, dataloader=None, save_dir=None, args=None, names=None, kpt_shape=None):
        super().__init__(dataloader, save_dir, args)
        self.names = names if names else {0: 'person'}
        self.nc = len(self.names)
        
        # 姿态专属属性
        self.kpt_shape = kpt_shape if kpt_shape else [17, 3]
        self.nkpt = self.kpt_shape[0]
        
        # COCO 格式 OKS 计算所需的各个关键点容差常数 (Sigma)
        self.sigma = np.array(
            [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]
        ) / 10.0
        
        self.metrics = None
        self.stats = {}

    def get_dataloader(self, dataset_path, batch_size=16):
        """
        验证集专属数据加载逻辑
        传递 task='pose' 标识以触发数据后端生成 OKS 计算所需的完整关键点字段
        """
        return create_dataloader(
            path=dataset_path,
            imgsz=getattr(self.args, 'imgsz', 640),
            batch_size=batch_size,
            task='pose',  
            is_training=False,
            num_workers=getattr(self.args, 'workers', 8)
        )

    def init_metrics(self, model):
        """初始化性能评估指标容器"""
        self.metrics = PoseMetrics(names=self.names)

    def preprocess(self, batch):
        """
        数据预处理阶段
        """
        for k in ["img", "image"]:
            if k in batch:
                input_tensor = batch[k]
                batch[k] = ops.cast(input_tensor, ms.float32) / 255.0
        return batch

    def postprocess(self, preds):
        """
        后处理逻辑：执行维度对齐与非极大值抑制 (NMS)。
        """
        p = preds[0] if isinstance(preds, (tuple, list)) else preds

        # 维度翻转适配：当特征维度小于锚框维度时，转置为 NMS 期望输入的 [B, anchors, features]
        if len(p.shape) == 3 and p.shape[1] < p.shape[2]:
            p = p.swapaxes(1, 2)

        conf_thres = getattr(self.args, 'conf', 0.001)
        iou_thres = getattr(self.args, 'iou', 0.6)

        out = non_max_suppression(
            p,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=getattr(self, 'nc', 1) 
        )
        return out

    def update_metrics(self, preds, batch):
        """基于模型预测输出与真实标签，计算并统计当前 Batch 的 IoU 与 OKS 匹配结果"""
        batch_idx = batch["batch_idx"].view(-1).asnumpy()
        gt_cls = batch["cls"].view(-1).asnumpy()
        gt_bboxes = batch["bboxes"].asnumpy().copy()
        gt_keypoints = batch["keypoints"].asnumpy().copy()

        imgsz = getattr(self.args, 'imgsz', 640)
        
        # --- 数据格式统一与坐标系对齐 ---
        # 尺度还原：若检测到真实边界框为 [0, 1] 归一化坐标，则放大至绝对像素尺度
        valid_boxes = gt_bboxes[gt_bboxes > 0] 
        if len(valid_boxes) > 0 and valid_boxes.max() <= 1.01:
            gt_bboxes[:, 0:4] *= imgsz
            gt_keypoints[..., 0:2] *= imgsz
        
        # 格式转换：统一将目标框坐标转换为绝对像素级的 [x1, y1, x2, y2] 格式
        labels_box_xyxy = xywh2xyxy_np(gt_bboxes) 
        
        # 边界约束：执行物理越界裁剪
        labels_box_xyxy[:, [0, 2]] = np.clip(labels_box_xyxy[:, [0, 2]], 0, imgsz)
        labels_box_xyxy[:, [1, 3]] = np.clip(labels_box_xyxy[:, [1, 3]], 0, imgsz)

        # 逐图执行指标评估统计
        for i, pred in enumerate(preds):
            idx_mask = (batch_idx == i) 
            labels_cls = gt_cls[idx_mask]
            labels_box = labels_box_xyxy[idx_mask]    
            labels_kpt = gt_keypoints[idx_mask]

            pred_np = pred.asnumpy() if len(pred) > 0 else np.zeros((0, 6 + self.nkpt * 3))

            # 漏检处理
            if len(pred_np) == 0:
                if len(labels_cls) > 0:
                    self.metrics.update_stats(
                        tp_b=np.zeros((0, 10), dtype=bool),
                        tp_p=np.zeros((0, 10), dtype=bool),
                        conf=np.zeros(0), pred_cls=np.zeros(0), target_cls=labels_cls
                    )
                continue

            p_boxes = pred_np[:, :4]  
            p_conf = pred_np[:, 4]    
            p_cls = pred_np[:, 5]     
            p_kpts = pred_np[:, 6:].reshape(-1, self.nkpt, 3) 

            tp_b = np.zeros((len(pred_np), 10), dtype=bool)
            tp_p = np.zeros((len(pred_np), 10), dtype=bool)

            if len(labels_box) > 0:
                # [A] 边界框 IoU 匹配计算
                iou_b = self.box_iou_np(p_boxes, labels_box)
                tp_b = self.match_predictions(p_cls, labels_cls, iou_b)

                # [B] 关键点 OKS 匹配计算
                # 依据目标框计算近似包围面积
                w = labels_box[:, 2] - labels_box[:, 0]
                h = labels_box[:, 3] - labels_box[:, 1]
                area = (w * h) * 0.53
                
                if len(p_kpts) > 0 and len(labels_kpt) > 0:
                    iou_p = kpt_iou(
                        ms.Tensor(p_kpts), 
                        ms.Tensor(labels_kpt), 
                        ms.Tensor(area), 
                        self.sigma
                    ).asnumpy()
                    tp_p = self.match_predictions(p_cls, labels_cls, iou_p)

            # 更新当前迭代批次的评估状态
            self.metrics.update_stats(tp_b, tp_p, p_conf, p_cls, labels_cls)

    def finalize_metrics(self):
        """验证循环终止阶段：计算评估指标（如 mAP）的最终结果"""
        if len(self.metrics.stats) == 0:
            print("[Warning] 验证过程未收集到有效目标数据，mAP 返回默认值 0.0。")
            self.stats = {"metrics/mAP50-95(B)": 0.0, "metrics/mAP50-95(P)": 0.0}
            return
            
        try:
            self.stats = self.metrics.process()
        except Exception as e:
            print(f"[Error] 评估指标计算过程发生异常: {e}")
            self.stats = {"metrics/mAP50-95(B)": 0.0, "metrics/mAP50-95(P)": 0.0}

    def get_stats(self):
        """向 Trainer 反馈格式化后的各项验证统计数据"""
        stats = super().get_stats()
        stats.update(self.stats)
        
        if self.metrics is not None:
            stats['fitness'] = self.metrics.fitness
        else:
            stats['fitness'] = 0.0
        self.results_dict = stats
        return stats

    def box_iou_np(self, box1, box2, eps=1e-7):
        """利用 NumPy 高效计算预测框与真实框之间的交并比 (IoU) 矩阵"""
        b1_x1, b1_y1, b1_x2, b1_y2 = np.split(box1, 4, axis=1)
        b2_x1, b2_y1, b2_x2, b2_y2 = np.split(box2, 4, axis=1)

        inter_area = (np.minimum(b1_x2, b2_x2.T) - np.maximum(b1_x1, b2_x1.T)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2.T) - np.maximum(b1_y1, b2_y1.T)).clip(0)

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter_area / (area1 + area2.T - inter_area + eps)

    def match_predictions(self, pred_cls, target_cls, iou, iou_thresholds=None):
        """采用贪心匹配算法，求解预测框与真实目标类别间的最佳一对一映射"""
        if iou_thresholds is None:
            iou_thresholds = np.linspace(0.5, 0.95, 10)

        tp = np.zeros((pred_cls.shape[0], iou_thresholds.shape[0]), dtype=bool)
        correct_class = pred_cls[:, None] == target_cls[None, :]
        iou = iou * correct_class

        for j, thr in enumerate(iou_thresholds):
            matches = np.argwhere(iou >= thr)
            if matches.shape[0] == 0:
                continue

            match_scores = iou[matches[:, 0], matches[:, 1]]
            match_data = np.concatenate((matches, match_scores[:, None]), axis=1)
            
            # 按匹配分数降序排序并执行去重操作
            match_data = match_data[match_data[:, 2].argsort()[::-1]]
            _, unique_preds = np.unique(match_data[:, 0], return_index=True)
            match_data = match_data[unique_preds]
            
            match_data = match_data[match_data[:, 2].argsort()[::-1]]
            _, unique_gts = np.unique(match_data[:, 1], return_index=True)
            match_data = match_data[unique_gts]

            tp[match_data[:, 0].astype(int), j] = True

        return tp