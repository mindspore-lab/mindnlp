import math
import mindspore.nn as nn

def get_lr(args, hyp, steps_per_epoch):
    """
    生成带有线性预热 (Linear Warmup) 与余弦退火 (Cosine Annealing) 的学习率调度序列
    
    Args:
        args: 包含全局训练轮数 (epochs) 等宏观配置的参数对象
        hyp (dict): 包含学习率、预热轮数等微观调优配置的超参数字典
        steps_per_epoch (int): 每个 Epoch 包含的迭代步数
        
    Returns:
        list[float]: 按步长展开的学习率列表
    """
    total_steps = args.epochs * steps_per_epoch
    
    # 从超参数字典中安全提取参数，提供默认值作为工程兜底
    warmup_epochs = hyp.get('warmup_epochs', 3.0)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    
    base_lr = hyp.get('lr0', 0.01)
    lrf = hyp.get('lrf', 0.01)  # 最终学习率比例
    min_lr = base_lr * lrf
    
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            # 预热阶段：从极小值线性增长至初始学习率 base_lr
            lr = base_lr * (i + 1) / warmup_steps
        else:
            # 余弦退火阶段
            cur_step = i - warmup_steps
            total_decay_steps = total_steps - warmup_steps
            
            # 使用标准的余弦退火公式计算当前步的学习率
            decay_ratio = 0.5 * (1 + math.cos(math.pi * cur_step / total_decay_steps))
            lr = min_lr + (base_lr - min_lr) * decay_ratio
            
        lr_each_step.append(lr)
        
    return lr_each_step

def build_optimizer(model, lr_list, hyp):
    """
    基于参数分组与配置字典构建优化器
    
    对一维张量（如 BatchNorm 权重）和偏置项（Bias）取消权重衰减（Weight Decay），
    以防限制模型的拟合能力或导致梯度异常
    
    Args:
        model (nn.Cell): 待优化的网络模型
        lr_list (list[float]): 预先计算好的学习率步长序列
        hyp (dict): 超参数字典，需包含 'optimizer', 'weight_decay', 'momentum' 等键
        
    Returns:
        nn.Optimizer: 实例化后的 MindSpore 优化器
    """
    decay_params = []
    no_decay_params = []
    
    # 执行参数分组：过滤不需要 L2 正则化的参数
    for param in model.trainable_params():
        if len(param.shape) == 1 or param.name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    weight_decay = hyp.get('weight_decay', 0.0005)
    
    group_params = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    # 动态解析并构建指定的优化器
    opt_name = hyp.get('optimizer', 'SGD').upper()
    momentum = hyp.get('momentum', 0.937)
    
    if opt_name == 'SGD':
        # YOLO 官方推荐默认使用带 Nesterov 动量的 SGD
        optimizer = nn.SGD(group_params, learning_rate=lr_list, momentum=momentum, nesterov=True)
    elif opt_name in ['ADAM', 'ADAMW']:
        # 微调或特定任务下使用的 AdamW
        optimizer = nn.AdamWeightDecay(group_params, learning_rate=lr_list, beta1=momentum, beta2=0.999)
    else:
        raise ValueError(f"[ERROR] 不支持的优化器类型: {opt_name}。请在 hyp.yaml 中指定 SGD 或 AdamW。")
        
    return optimizer