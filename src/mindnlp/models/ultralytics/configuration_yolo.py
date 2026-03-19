import os
import yaml

#from mindnlp.models.utils import PretrainedConfig
# 尝试获取 MindNLP 基类，增强套件兼容性

try:

    from mindnlp.models.utils import PretrainedConfig

except Exception:

    # 兼容性防御：如果环境未完全安装，使用基础 Config 类防止崩溃

    class PretrainedConfig:

        def __init__(self, **kwargs):

            for k, v in kwargs.items():

                setattr(self, k, v)

class YOLOConfig(PretrainedConfig):
    """
    YOLO11 全任务通用配置类
    支持：分类 (Classify)、检测 (Detect)、分割 (Segment)、姿态 (Pose)
    """
    model_type = "yolo11"

    def __init__(
        self,
        yaml_path=None,      
        scale='n',           
        nc=80,               
        kpt_shape=None,      
        reg_max=16,          
        nm=32,               
        npr=256,             
        **kwargs
    ):
        # 修复 Python 可变默认参数陷阱
        if kpt_shape is None:
            kpt_shape = [17, 3]
            
        self.yaml_path = yaml_path
        self.scale = scale
        self.reg_max = reg_max
        self.nm = nm
        self.npr = npr
        self.yaml_dict = {}
        
        # 先设置基础兜底参数
        self.nc = nc
        self.kpt_shape = kpt_shape
        self.depth_multiple = 1.0
        self.width_multiple = 1.0
        self.max_channels = 1024
        self.backbone = []
        self.head = []
        
        # 读取并解析 YAML
        if yaml_path:
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    self.yaml_dict = yaml.safe_load(f)
            else:
                raise FileNotFoundError(f"错误：找不到 YAML 配置文件 '{yaml_path}'。请检查相对路径！")
        
        # 从 YAML 中覆盖参数
        if self.yaml_dict:
            self.nc = self.yaml_dict.get('nc', self.nc)
            self.kpt_shape = self.yaml_dict.get('kpt_shape', self.kpt_shape)
            self.backbone = self.yaml_dict.get('backbone', [])
            self.head = self.yaml_dict.get('head', [])
            
            # 根据 scale 动态解析 depth, width, max_channels
            scales = self.yaml_dict.get('scales', {})
            if scale in scales:
                self.depth_multiple, self.width_multiple, self.max_channels = scales[scale]
            else:
                print(f"警告: YAML 中未找到 scale '{scale}'，将使用默认值 1.0")
                
        super().__init__(**kwargs)