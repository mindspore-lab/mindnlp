import os
import argparse
import numpy as np
import mindspore as ms
from mindspore import Tensor
from ultralytics import YOLO

import modeling_yolo
from configuration_yolo import YOLOConfig

def translate_ms_to_pt(ms_name, task):
    """
    参数映射路由表：将 MindSpore 架构下的参数名称转换为 PyTorch 架构对应的参数名称
    """
    #  基础规范化转换：对齐 BatchNorm 以及权重、偏置的命名差异
    pt_name = ms_name.replace(".moving_mean", ".running_mean") \
                     .replace(".moving_variance", ".running_var") \
                     .replace(".gamma", ".weight") \
                     .replace(".beta", ".bias")
                     
    #  结构命名转换：对齐 C2f/Bottleneck 等内部核心模块的命名规范
    pt_name = pt_name.replace(".conv1.", ".cv1.")
    pt_name = pt_name.replace(".conv2.", ".cv2.")
    pt_name = pt_name.replace(".conv3.", ".cv3.")
    
    return pt_name

def universal_convert(task="segment", scale="n"):
    """
    通用权重转换流水线。
    支持将 Ultralytics 官方 PyTorch (.pt) 权重转换为符合 MindNLP 规范的 MindSpore (.ckpt) 权重
    """
    # 1. 初始化任务及配置映射字典
    model_map = {
        "classify": modeling_yolo.YOLO11ForClassification,
        "detect": modeling_yolo.YOLO11ForObjectDetection,
        "segment": modeling_yolo.YOLO11ForSegmentation,
        "pose": modeling_yolo.YOLO11ForPose
    }
    pt_name_map = {
        "classify": f"yolo11{scale}-cls.pt", 
        "detect": f"yolo11{scale}.pt", 
        "segment": f"yolo11{scale}-seg.pt",
        "pose": f"yolo11{scale}-pose.pt"           
    }
    yaml_map = {
        "classify": "cfg/models/11/yolo11-cls.yaml", 
        "detect": "cfg/models/11/yolo11.yaml", 
        "segment": "cfg/models/11/yolo11-seg.yaml",
        "pose": "cfg/models/11/yolo11-pose.yaml"
    }
    
    nc_map = {"classify": 1000, "detect": 80, "segment": 80, "pose": 1} 

    print(f"[INFO] 启动 YOLO11-{task.upper()} 权重转换流程...")
    
    current_yaml = yaml_map[task]
    
    # 2. 动态构建网络配置与 MindSpore 模型实例
    if task == "pose":
        cfg = YOLOConfig(yaml_path=current_yaml, scale=scale, nc=nc_map[task], kpt_shape=[17, 3])
    else:
        cfg = YOLOConfig(yaml_path=current_yaml, scale=scale, nc=nc_map[task])
        
    ms_model = model_map[task](cfg)
    ms_model.set_train(False)

    # 3. 加载 PyTorch 原生权重
    print(f"[INFO] 正在解析 PyTorch 权重文件: {pt_name_map[task]}")
    pt_yolo = YOLO(pt_name_map[task])
    pt_dict = pt_yolo.model.state_dict()

    print("[DEBUG] 官方 PyTorch 权重键名抽样 (末尾 5 项):")
    pt_keys = list(pt_dict.keys())
    for k in pt_keys[-5:]:
        print(f"  {k:50} | 尺寸: {list(pt_dict[k].shape)}")
    print("-" * 80)
    
    new_ms_ckpt = []
    matched_count = 0
    ms_params = list(ms_model.parameters_and_names())
    
    print("[INFO] 开始执行参数映射与维度校验...")
    
    # 4. 执行转换主循环
    for ms_name, ms_param in ms_params:
        ms_shape = tuple(ms_param.shape)
        ms_size = ms_param.size
        
        # 获取期望映射的 PyTorch 键名
        expected_pt_name = translate_ms_to_pt(ms_name, task)

        if expected_pt_name in pt_dict:
            pt_v = pt_dict[expected_pt_name]
            
            # 维度一致性校验
            if pt_v.numel() == ms_size:
                val_np = pt_v.cpu().numpy().reshape(ms_shape)
                
                # 安全策略：限制 BatchNorm 历史方差下限，防止半精度推理时出现除零溢出
                if "moving_variance" in ms_name:
                    val_np = np.maximum(val_np, 1e-5)
                    
                ms_param.set_data(Tensor(val_np, ms.float32))
                new_ms_ckpt.append({'name': ms_name, 'data': ms_param.data})
                
                print(f"[对齐成功] {ms_name:<55} <- {expected_pt_name}")
                matched_count += 1
                
                # 内存优化：匹配成功后从字典中移除该键，以便最后审计遗留项
                del pt_dict[expected_pt_name] 
            else:
                print(f"[形状冲突] 参数 {ms_name} 期望尺寸 {ms_shape}，实际载入尺寸 {tuple(pt_v.shape)}")
        else:
            # 处理常量和不可训练张量 (如 DFL 积分权重、步长锚点)
            # 由于网络初始化时已经通过 Config 构建了正确的值，此处直接保留并存入 Checkpoint 即可
            if "stride" in ms_name or "dfl.conv.weight" in ms_name:
                new_ms_ckpt.append({'name': ms_name, 'data': ms_param.data})
                print(f"[保留原值] {ms_name:<55} (框架内置固定张量)")
                matched_count += 1
            else:
                # 若非常量且未能在 PT 字典中找到映射，留交审计模块处理
                pass

    # 5. 输出转换审计报告
    print("-" * 80)
    print("[INFO] 权重转换审计清单")
    matched_names = [x['name'] for x in new_ms_ckpt]
    failed_names = [n for n, _ in ms_params if n not in matched_names]
    
    if not failed_names: 
        print("  校验通过：所有模型参数均已成功映射。")
    else: 
        print(f"  校验警告：存在 {len(failed_names)} 个未匹配参数，请检查拓扑结构：")
        for fn in failed_names: 
            print(f"    - {fn}")

    # 6. 持久化存储
    base_name = os.path.splitext(pt_name_map[task])[0]
    save_ckpt_path = f"{base_name}.ckpt"

    ms.save_checkpoint(new_ms_ckpt, save_ckpt_path)
    print(f"[INFO] 转换结束。参数对齐率: {matched_count}/{len(ms_params)}。已序列化至: {save_ckpt_path}")
    
    return save_ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 参数转换工具 (PyTorch to MindSpore Checkpoint)")
    parser.add_argument("--task", "-t", type=str, default="pose", 
                        choices=["classify", "detect", "segment", "pose"],
                        help="目标模型的基础任务类型")
    parser.add_argument("--scale", "-s", type=str, default="n", 
                        choices=["n", "s", "m", "l", "x"],
                        help="指定模型的规模缩放因子") 
    args = parser.parse_args()

    universal_convert(args.task, args.scale)