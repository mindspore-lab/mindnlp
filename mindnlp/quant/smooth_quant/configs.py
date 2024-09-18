"""quant configs"""
def no(model_cfg, act_max):
    return {}


# 静态混合精度分解
def sd(model_cfg, act_max):
    quant_cfg = {}
    h_mx, d_mx = findN(0.04 * model_cfg.hidden_size), findN(
        0.1 * model_cfg.intermediate_size
    )
    scale, step = 4, 4 / model_cfg.num_hidden_layers
    for i in range(model_cfg.num_hidden_layers):
        scale = max(0, scale - step)
        h_cur, d_cur = max(16, h_mx >> int(scale)), max(32, d_mx >> int(scale))
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]:
            quant_cfg[str(i) + "." + name] = {
                "type": "W8SD",
                "act_scale": True,
                "alpha": h_cur,
            }
        quant_cfg[str(i) + ".down_proj"] = {
            "type": "W8SD",
            "act_scale": True,
            "alpha": d_cur,
        }
    quant_cfg["lm_head"] = {"type": "W8SD"}
    quant_cfg["act_scales_path"] = act_max
    return quant_cfg


def findN(N):
    sum = 1
    while True:
        if sum * 2 > N:
            return sum
        sum = sum * 2


# 平滑激活
def smooth(model_cfg, act_max):
    quant_cfg = {}
    for i in range(model_cfg.num_hidden_layers):
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]:
            quant_cfg[str(i) + "." + name] = {"type": "W8X8"}
        # 对某一个具体的层加act_scale的作用： 若为W8X8，则对该层进行smooth；如为W8SD，则用act_scale进行混合精度分解。
        quant_cfg[str(i) + ".down_proj"] = {
            "type": "W8X8",
            "act_scale": True,
            "alpha": 0.85,
        }
    quant_cfg["lm_head"] = {"type": "W8X8", "act_scale": True, "alpha": 0.85}
    quant_cfg["act_scales_path"] = act_max
    quant_cfg["alpha"] = 0.85  # smoothquant 迁移系数
    quant_cfg["smooth"] = (
        True  # 整体的smooth控制是将激活值的缩放与RMSNorm融合，不会造成额外的开销，但down_proj层无法使用
    )
    return quant_cfg


# 对down_proj混合精度分解，对其他部分平滑激活
def smsd(model_cfg, act_max):
    quant_cfg = {}
    d_mx = findN(0.1 * model_cfg.intermediate_size)
    scale, step = 4, 4 / model_cfg.num_hidden_layers
    for i in range(model_cfg.num_hidden_layers):
        scale = max(0, scale - step)
        d_cur = max(32, d_mx >> int(scale))
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]:
            quant_cfg[str(i) + "." + name] = {"type": "W8X8"}
        quant_cfg[str(i) + ".down_proj"] = {
            "type": "W8SD",
            "act_scale": True,
            "alpha": d_cur,
        }
    quant_cfg["lm_head"] = {"type": "W8SD", "act_scale": True, "alpha": 64}
    quant_cfg["act_scales_path"] = act_max
    quant_cfg["smooth"] = True
    return quant_cfg


# 仅权重int8量化
def w8(model_cfg, act_max):
    quant_cfg = {}
    for i in range(model_cfg.num_hidden_layers):
        for name in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]:
            quant_cfg[str(i) + "." + name] = {"type": "W8"}
    quant_cfg["lm_head"] = {"type": "W8"}
    return quant_cfg


# 动态混合精度分解
def w8dx(model_cfg, act_max):
    quant_cfg = {}
    for i in range(model_cfg.num_hidden_layers):
        for name in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]:
            quant_cfg[str(i) + "." + name] = {"type": "W8DX"}
    # quant_cfg["lm_head"] = {"type":"W8DX"}  # 可以根据需要取消注释
    # quant_cfg["act_scales_path"] = act_max # 可以根据需要取消注释
    # quant_cfg["smooth"] = True # 可以根据需要取消注释
    return quant_cfg


# per-token absmax量化
def w8x8(model_cfg):
    quant_cfg = {}
    for i in range(model_cfg.num_hidden_layers):
        for name in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]:
            quant_cfg[str(i) + "." + name] = {"type": "W8X8"}
    quant_cfg["lm_head"] = {"type": "W8X8"}
    return quant_cfg
