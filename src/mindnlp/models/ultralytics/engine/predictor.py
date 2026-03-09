import os
import time
import logging
import yaml
import numpy as np
import cv2
import mindspore as ms
from mindspore import Tensor

# 统一日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

class BasePredictor:
    """
    MindNLP 推理引擎基类
    负责端到端的推理：数据加载 -> 预处理 -> 前向推理 -> 后处理
    """

    def __init__(self, cfg=None):
        """
        初始化预测器配置
        """
        self.cfg = cfg or {}
        
        # 解析超参数配置 (支持从 hyp.yaml 读取 conf, iou 等推理参数)
        self.hyp = {}
        if hasattr(self.cfg, 'hyp') and self.cfg.hyp and os.path.exists(self.cfg.hyp):
            with open(self.cfg.hyp, "r", encoding="utf-8") as f:
                self.hyp = yaml.safe_load(f)

        self.conf_thres = self.hyp.get('conf', getattr(self.cfg, 'conf', 0.25))
        self.iou_thres = self.hyp.get('iou', getattr(self.cfg, 'iou', 0.45))
        self.imgsz = getattr(self.cfg, 'imgsz', 640)

        self.model = None
        self.dataset = []  
        self.results = []

    def setup_model(self, model, ckpt_path=None):
        """加载模型权重并设置推理模式"""
        self.model = model
        if ckpt_path and os.path.exists(ckpt_path):
            param_dict = ms.load_checkpoint(ckpt_path)
            ms.load_param_into_net(self.model, param_dict, strict_load=False)
            LOGGER.info(f"成功加载检查点权重: {ckpt_path}")

        # 锁定为评估模式，冻结 BatchNorm 等动态层
        self.model.set_train(False)

    def setup_source(self, source):
        """解析输入数据源 (支持单张图像或目录遍历)"""
        self.dataset = []
        if isinstance(source, str):
            if os.path.isdir(source):
                for f in os.listdir(source):
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.dataset.append(os.path.join(source, f))
            elif os.path.isfile(source):
                self.dataset.append(source)
        elif isinstance(source, np.ndarray):
            self.dataset.append(source) 

        if not self.dataset:
            raise ValueError(f"[ERROR] 无法从指定数据源解析到有效图像: {source}")

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """
        图像仿射变换：等比例缩放并对边缘进行 Padding 填充
        返回处理后的图像以及变换参数 (供坐标还原使用)
        """
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
        dw, dh = dw / 2, dh / 2

        if shape[::-1] != new_unpad:  
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, (r, dw, dh)

    def preprocess(self, img_source):
        """
        基类图像预处理流水线：BGR -> RGB -> LetterBox -> CHW -> Normalize -> Tensor
        """
        if isinstance(img_source, str):
            orig_img = cv2.imread(img_source)
        else:
            orig_img = img_source.copy()

        img, (ratio, pad_w, pad_h) = self.letterbox(orig_img, new_shape=(self.imgsz, self.imgsz))

        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)

        im_tensor = Tensor(img, ms.float32) / 255.0
        im_tensor = im_tensor.expand_dims(0) 

        return im_tensor, orig_img, (ratio, pad_w, pad_h)

    def inference(self, im_tensor):
        """执行模型前向传播"""
        return self.model(im_tensor)

    def postprocess(self, preds, orig_img, preprocess_info):
        """后处理抽象方法，需由各任务子类具体实现"""
        raise NotImplementedError("BasePredictor 不执行特定的解析逻辑，请在子类中重写 postprocess 方法。")

    def __call__(self, source, model=None, ckpt_path=None):
        """推理流水线主调度入口"""
        if model is not None:
            self.setup_model(model, ckpt_path)

        if self.model is None:
            raise RuntimeError("[ERROR] 模型未初始化，无法启动推理流水线。")

        self.setup_source(source)
        self.results = []

        LOGGER.info(f"推理引擎启动，共探测到 {len(self.dataset)} 份输入样本。")

        for img_path_or_arr in self.dataset:
            # 1. 预处理
            t1 = time.time()
            im_tensor, orig_img, prep_info = self.preprocess(img_path_or_arr)

            # 2. 推理
            t2 = time.time()
            preds = self.inference(im_tensor)

            # 3. 后处理
            t3 = time.time()
            result = self.postprocess(preds, orig_img, prep_info)
            self.results.append(result)

            # 4. 耗时统计
            inf_time = (t3 - t2) * 1000
            post_time = (time.time() - t3) * 1000
            name = img_path_or_arr if isinstance(img_path_or_arr, str) else "numpy_array"
            LOGGER.info(f"处理完成 [{os.path.basename(name)}] | 前向推理: {inf_time:.1f}ms | 后处理: {post_time:.1f}ms")

        return self.results