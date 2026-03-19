import math
import numpy as np
import mindspore as ms
from mindspore import nn, ops

#from mindnlp.core.nn import PreTrainedModel 

try:
    # 尝试导入官方的预训练基类
    from mindnlp.core.nn import PreTrainedModel 
except ImportError as e:
    print(f"⚠️ 警告: 无法从 mindnlp 导入 PreTrainedModel (可能是 mindtorch 版本冲突)。启动降级模式。错误: {e}")
    # 本地降级方案：用一个假的基类糊弄过去，保证 train.py 能正常跑
    class PreTrainedModel(nn.Cell):
        config_class = None
        def __init__(self, config):
            super().__init__()
            self.config = config

from configuration_yolo import YOLOConfig
from modules import ConvNormAct, C3k2, C2PSA, Classify, YOLO11DetectHead, YOLO11Segment, YOLO11Pose, Concat, Identity, SPPF, Upsample

# 模块映射字典

MODULE_MAP = {
    'Conv': ConvNormAct,
    'C3k2': C3k2,
    'C2PSA': C2PSA,
    'SPPF': SPPF,
    'Concat': Concat,
    'nn.Upsample': Upsample,
    'Identity': Identity,
    'Detect': YOLO11DetectHead,
    'Segment': YOLO11Segment,
    'Pose': YOLO11Pose,
    'Classify': Classify
}

class YOLO11Base(PreTrainedModel):
    """
    YOLO11 通用基类，负责解析 YAML 配置、搭建网络拓扑结构以及权重的初始化。
    """
    
    # 绑定配置类，使得 from_pretrained 能够自动解析对应的 Config
    config_class = YOLOConfig 
    
    def __init__(self, config):
        super().__init__(config) 
        self.config = config
        self.nc = config.nc
        
        # 动态解析网络结构
        self.model, self.save = self._parse_model(config)
        self._init_weights()
        self._init_logits_weights()

    def _init_weights(self):
        """统一的权重初始化，包含针对 DFL 层的特殊跳过逻辑"""
        for name, cell in self.cells_and_names():
            # DFL (Distribution Focal Loss) 层的权重是固定的预设值，不能被随机初始化覆盖
            if "dfl" in name.lower():
                continue
                
            if isinstance(cell, nn.Conv2d):
                # 使用 HeUniform 初始化卷积层，适配 ReLU/SiLU 激活函数
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer('ones', cell.gamma.shape))
                cell.beta.set_data(ms.common.initializer.initializer('zeros', cell.beta.shape))

    def _init_logits_weights(self):
        """
        初始化分类/检测头的偏置项
        防止网络训练初期 Loss 爆炸
        """
        head = self.model[-1] if hasattr(self, 'model') else None
        
        if head and hasattr(head, 'cv3'): 
            for conv_seq in head.cv3:
                last_conv = None
                # 倒序查找包含 bias 的最后一层卷积
                cells = list(conv_seq.cells_and_names())
                for _, cell in reversed(cells):
                    if isinstance(cell, nn.Conv2d):
                        last_conv = cell
                        break
                
                if last_conv is not None and last_conv.has_bias:
                    # 使用 -4.59 填充 bias
                    new_bias = np.full(last_conv.bias.shape, -4.59)
                    last_conv.bias.set_data(ms.Tensor(new_bias, ms.float32))

    def _parse_model(self, config):
        """根据配置项动态生成网络层和拓扑结构"""
        gd = config.depth_multiple
        gw = config.width_multiple
        max_ch = getattr(config, 'max_channels', 1024) or 1024
        
        if not hasattr(config, 'yaml_dict'):
            raise ValueError("Config validation failed: `yaml_dict` is missing. Ensure YOLOConfig is loaded correctly.")
            
        layers_cfg = config.yaml_dict['backbone'] + config.yaml_dict['head']
        layers, ch, save = [], [3], []
        
        # 提取各个层的拓扑连接信息，避免在 construct 中动态读取 cell.f 和 cell.i
        self.layer_f = [] 
        self.layer_i = []

        for i, (f, n, m_name, args) in enumerate(layers_cfg):
            m = MODULE_MAP.get(m_name, m_name) if isinstance(m_name, str) else m_name
            n = max(round(n * gd), 1) if n > 1 else n
            
            if isinstance(f, list):
                c1 = [ch[x + 1 if x >= 0 else x] for x in f]
                save.extend(x for x in f if x != -1)
                
                if m in (YOLO11DetectHead, YOLO11Segment, YOLO11Pose):
                    _reg_max = 16
                    _stride = [8, 16, 32]
                    
                    if m is YOLO11DetectHead:
                        _reg_max = args[1] if len(args) > 1 and isinstance(args[1], int) else 16
                        m_ = m(nc=self.nc, reg_max=_reg_max, stride=_stride, ch=c1)
                        
                    elif m is YOLO11Segment:
                        _nm = args[1] if len(args) > 1 and isinstance(args[1], int) else 32
                        _npr = args[2] if len(args) > 2 and isinstance(args[2], int) else 256
                        _reg_max = args[3] if len(args) > 3 and isinstance(args[3], int) else 16
                        m_ = m(nc=self.nc, reg_max=_reg_max, nm=_nm, npr=_npr, stride=_stride, ch=c1)
                        
                    elif m is YOLO11Pose:
                        _kpt_shape = args[1] if len(args) > 1 and isinstance(args[1], (list, tuple)) else getattr(config, 'kpt_shape', [17, 3])
                        _reg_max = args[2] if len(args) > 2 and isinstance(args[2], int) else 16
                        m_ = m(nc=self.nc, kpt_shape=_kpt_shape, reg_max=_reg_max, stride=_stride, ch=c1)
                        
                    c2 = ch[-1]
                elif m is Concat:
                    m_ = m(*args)
                    c2 = sum(c1)
                else:
                    m_ = m(*args)
                    c2 = ch[-1]
            elif m is Upsample:
                m_ = m(size=args[0], scale_factor=args[1], mode=args[2])
                c2 = ch[-1]
            else:
                c1 = ch[f]
                if m is Classify:
                    m_ = m(c1, self.nc, *args)
                    c2 = self.nc
                else:
                    c2 = math.ceil(min(args[0], max_ch) * gw / 8) * 8
                    new_args = [c1, c2, n, *args[1:]] if m in (C3k2, C2PSA) else [c1, c2, *args[1:]]
                    m_ = m(*new_args) if n == 1 else nn.SequentialCell([m(*new_args) for _ in range(n)])

            # 使用列表记录拓扑关系，对静态图更友好
            self.layer_i.append(i)
            self.layer_f.append(f)
            
            layers.append(m_)
            ch.append(c2)
            
        return nn.CellList(layers), sorted(save)

    def construct(self, x):
        """前向传播逻辑"""
        y = []
        for i, cell in enumerate(self.model):
            f = self.layer_f[i]
            
            # 处理多输入或者跳连特征获取
            if f != -1:
                x = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            
            x = cell(x)
                
            y.append(x if self.layer_i[i] in self.save else None)
        return x

class YOLO11ForObjectDetection(YOLO11Base):
    def construct(self, x):
        res = super().construct(x)
        
        if self.training:
            # 训练模式
            return res
        
        # 推理模式
        x = res[0] if isinstance(res, (list, tuple)) else res
        if x.ndim == 2: 
            x = ops.expand_dims(x, 0)
        if x.shape[1] > x.shape[2]: 
            x = x.transpose(0, 2, 1)
        return x

class YOLO11ForSegmentation(YOLO11Base):
    """YOLO11 实例分割模型"""
    def construct(self, x):
        # 分割 Head 这里的 x 通常返回 (pred, proto)
        x = super().construct(x)
        if not self.training:
            pred, proto = x[0], x[1]
            if pred.shape[1] > pred.shape[2]: 
                pred = pred.transpose(0, 2, 1)
            return pred, proto
        return x

class YOLO11ForClassification(YOLO11Base):
    """YOLO11 图像分类模型"""
    def construct(self, x, labels=None):
        x = super().construct(x)
        if labels is not None:
            # 增加对 labels 的支持，返回 (Loss, Logits) 以兼容 Hugging Face/MindNLP Trainer
            loss = nn.CrossEntropyLoss()(x, labels)
            return (loss, x)
        return x

class YOLO11ForPose(YOLO11Base):
    """YOLO11 姿态估计模型"""
    def construct(self, x):
        x = super().construct(x)
        
        if not self.training:
            if isinstance(x, (list, tuple)):
                pred = x[0] # 取出主预测结果
            else:
                pred = x
                
            # 确保预测张量是 [Batch, Channels, Anchors]
            if pred.ndim == 3 and pred.shape[1] > pred.shape[2]: 
                pred = pred.transpose(0, 2, 1)
                
            # 包装成列表返回，适配官方 Validator
            return [pred]
            
        return x