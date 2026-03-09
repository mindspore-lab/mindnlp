import math
import copy
import mindspore as ms
from mindspore import ops

class ModelEMA:
    """ 
    MindSpore版 ModelEMA
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # 深度拷贝模型实例
        self.ema_model = copy.deepcopy(model)
        self.ema_model.set_train(False)
        self.updates = updates
        
        # 动态衰减率策略
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        
        # 提取参数列表 (转为字典以防顺序错乱)
        self.ema_params = {p.name: p for p in self.ema_model.get_parameters()}
        self.model_params = {p.name: p for p in model.get_parameters()}
        
        # 锁定梯度
        for param in self.ema_params.values():
            param.requires_grad = False

    def update(self, model):
        """ 执行单步 EMA 更新 """
        self.updates += 1
        d = self.decay(self.updates)
        
        # 提取当前最新模型参数
        current_model_params = {p.name: p for p in model.get_parameters()}
        
        for name, ema_p in self.ema_params.items():
            model_p = current_model_params.get(name)
            if model_p is None:
                continue
                
            if model_p.dtype in [ms.float16, ms.float32]:
                if "moving_mean" in name or "moving_variance" in name:
                    ema_p.set_data(model_p.value())
                else:
                    # 常规权重执行 EMA 平滑
                    new_val = d * ema_p.value() + (1.0 - d) * model_p.value()
                    ema_p.set_data(new_val)