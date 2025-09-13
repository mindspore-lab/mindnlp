# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
MoE路由机制演示

本示例展示了混合专家（MoE）模型中的路由机制如何工作，包括：
1. 路由器如何为每个token选择专家
2. 负载均衡的重要性
3. 不同路由策略的效果
"""

import mindspore
from mindspore import nn, ops, context, Tensor
import numpy as np
import matplotlib.pyplot as plt

# 设置动态图模式
context.set_context(mode=context.PYNATIVE_MODE)


class SimpleRouter(nn.Cell):
    """简单的路由器实现"""
    
    def __init__(self, input_dim, num_experts, add_noise=False):
        super().__init__()
        self.gate = nn.Dense(input_dim, num_experts, has_bias=False)
        self.add_noise = add_noise
        
    def construct(self, x):
        # 计算每个专家的分数
        logits = self.gate(x)
        
        # 可选：添加噪声以增加探索性
        if self.add_noise and self.training:
            noise = ops.randn_like(logits) * 0.1
            logits = logits + noise
            
        return logits


class LoadBalancedRouter(nn.Cell):
    """带负载均衡的路由器"""
    
    def __init__(self, input_dim, num_experts, capacity_factor=1.5):
        super().__init__()
        self.gate = nn.Dense(input_dim, num_experts, has_bias=False)
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        
    def construct(self, x, return_aux_loss=False):
        # 处理不同的输入形状
        if x.ndim == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = x.reshape(-1, hidden_dim)
        elif x.ndim == 2:
            # 已经是扁平化的输入
            x_flat = x
            batch_size, seq_len = 1, x.shape[0]
            hidden_dim = x.shape[1]
        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")
        
        total_tokens = batch_size * seq_len
        
        # 计算路由分数
        logits = self.gate(x_flat)
        
        # 计算每个专家的负载（用于辅助损失）
        probs = ops.softmax(logits, axis=-1)
        
        # 选择top-k专家
        routing_weights, selected_experts = ops.topk(logits, k=2)
        routing_weights = ops.softmax(routing_weights, axis=-1)
        
        if return_aux_loss:
            # 计算负载均衡损失
            # 理想情况下，每个专家应该处理相同数量的tokens
            tokens_per_expert = ops.zeros(self.num_experts)
            for i in range(self.num_experts):
                mask = (selected_experts == i).any(axis=-1).astype(mindspore.float32)
                tokens_per_expert[i] = mask.sum()
            
            # 计算每个专家的平均概率
            avg_probs_per_expert = probs.mean(axis=0)
            
            # 辅助损失：鼓励均匀分布
            ideal_load = total_tokens / self.num_experts
            load_balancing_loss = ops.square(tokens_per_expert - ideal_load).mean()
            
            return routing_weights, selected_experts, load_balancing_loss
        
        return routing_weights, selected_experts


def visualize_routing_patterns(router, inputs, title="Routing Pattern Visualization"):
    """可视化路由决策"""
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 确保输入形状正确
        if inputs.ndim == 3:
            batch_size, seq_len, hidden_dim = inputs.shape
            inputs_flat = inputs.reshape(-1, hidden_dim)  # 展平为2D
        else:
            inputs_flat = inputs
            batch_size, seq_len = 1, inputs.shape[0]
            hidden_dim = inputs.shape[1]
        
        print(f"处理输入: 原始形状={inputs.shape}, 展平后形状={inputs_flat.shape}")
        
        # 计算路由决策
        # 处理不同类型的路由器返回值
        if isinstance(router, LoadBalancedRouter):
            # LoadBalancedRouter可能返回多个值，我们只需要logits
            # 直接调用router的gate来获取logits
            logits = router.gate(inputs_flat)
        else:
            # 其他路由器直接返回logits
            logits = router(inputs_flat)
            
        probs = ops.softmax(logits, axis=-1)
        
        # 获取路由决策
        _, selected = ops.topk(logits, k=2)
            
        # 转换为numpy进行可视化
        probs_np = probs.asnumpy()  # shape: [num_tokens, num_experts]
        selected_np = selected.asnumpy()  # shape: [num_tokens, k]
        
        print(f"可视化数据形状: probs_np={probs_np.shape}, selected_np={selected_np.shape}")
        
        # 确保probs_np是2D的
        if probs_np.ndim != 2:
            print(f"错误：概率数组应该是2D的，但得到了{probs_np.ndim}D: {probs_np.shape}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 热力图显示所有专家的概率 - probs_np应该是2D的[tokens, experts]
        im1 = ax1.imshow(probs_np.T, aspect='auto', cmap='hot')  # 转置以便专家在Y轴
        ax1.set_xlabel('Token ID')
        ax1.set_ylabel('Expert ID')
        ax1.set_title(f'{title} - Expert Probability Distribution')
        plt.colorbar(im1, ax=ax1)
        
        # 显示被选中的专家
        num_experts = probs_np.shape[1]
        expert_counts = np.zeros(num_experts)
        
        # 统计每个专家被选中的次数
        for i in range(selected_np.shape[0]):  # 遍历所有tokens
            for j in range(selected_np.shape[1]):  # 遍历top-k选择
                expert_id = selected_np[i, j]
                if 0 <= expert_id < num_experts:
                    expert_counts[expert_id] += 1
                
        ax2.bar(range(len(expert_counts)), expert_counts)
        ax2.set_xlabel('Expert ID')
        ax2.set_ylabel('Selection Count')
        ax2.set_title(f'{title} - Expert Load Distribution')
        
        plt.tight_layout()
        
        # 保存到当前目录，文件名安全化
        safe_title = title.replace(" ", "_").replace("-", "_").replace(":", "_").replace("/", "_")
        save_path = f'{safe_title}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存为: {save_path}")
        
        # 在Windows环境下，显示可能有问题，所以只保存不显示
        try:
            plt.show()
        except Exception as e:
            print(f"显示图片时出错: {e}")
        
        plt.close()  # 关闭图片释放内存
        
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        print(f"输入形状: {inputs.shape}")
        import traceback
        traceback.print_exc()


def demonstrate_routing_strategies():
    """演示不同的路由策略"""
    print("="*60)
    print("MoE路由机制演示")
    print("="*60)
    
    # 参数设置
    batch_size = 4
    seq_len = 16
    hidden_dim = 128
    num_experts = 8
    
    # 创建输入数据
    # 模拟不同类型的输入以测试路由
    inputs = []
    
    # 类型1：正态分布
    inputs.append(ops.randn(batch_size, seq_len, hidden_dim))
    
    # 类型2：带有明显模式的输入
    pattern_input = ops.zeros((batch_size, seq_len, hidden_dim))
    for i in range(seq_len):
        pattern_input[:, i, i % hidden_dim] = 5.0
    inputs.append(pattern_input)
    
    # 类型3：稀疏输入
    sparse_input = ops.randn(batch_size, seq_len, hidden_dim)
    mask = ops.rand(batch_size, seq_len, hidden_dim) > 0.8
    sparse_input = sparse_input * mask.astype(mindspore.float32)
    inputs.append(sparse_input)
    
    input_names = ["Random Input", "Pattern Input", "Sparse Input"]
    
    # 测试不同的路由器
    routers = [
        ("Simple Router", SimpleRouter(hidden_dim, num_experts)),
        ("Noisy Router", SimpleRouter(hidden_dim, num_experts, add_noise=True)),
        ("Load Balanced Router", LoadBalancedRouter(hidden_dim, num_experts))
    ]
    
    for router_name, router in routers:
        print(f"\n{router_name}:")
        print("-" * 40)
        
        for input_data, input_name in zip(inputs, input_names):
            print(f"\n输入类型: {input_name}")
            
            if isinstance(router, LoadBalancedRouter):
                weights, experts, aux_loss = router(input_data, return_aux_loss=True)
                print(f"  负载均衡损失: {aux_loss.item():.4f}")
            else:
                router_output = router(input_data)
                weights, experts = ops.topk(router_output.reshape(-1, num_experts), k=2)
                weights = ops.softmax(weights, axis=-1)
            
            # 分析路由分布
            expert_usage = ops.zeros(num_experts)
            for i in range(num_experts):
                usage = (experts == i).sum()
                expert_usage[i] = usage
                
            print(f"  专家使用分布: {expert_usage.asnumpy()}")
            print(f"  最常用专家: {expert_usage.argmax().item()}")
            print(f"  最少用专家: {expert_usage.argmin().item()}")
            print(f"  使用率标准差: {expert_usage.std().item():.2f}")
            
            # 可视化第一个输入的路由模式
            if input_name == "Random Input":
                visualize_routing_patterns(
                    router, 
                    input_data[0:1],  # 只用第一个batch
                    f"{router_name}-{input_name}"
                )


def analyze_capacity_constraints():
    """分析容量限制对路由的影响"""
    print("\n\n容量限制分析")
    print("="*60)
    
    hidden_dim = 128
    num_experts = 8
    seq_len = 100  # 长序列
    
    # 创建偏斜的输入（某些token更倾向于特定专家）
    input_data = ops.randn(1, seq_len, hidden_dim)
    # 添加偏斜
    for i in range(0, seq_len, 10):
        input_data[:, i:i+5, :] += ops.randn(1, 5, hidden_dim) * 2
    
    # 不同容量因子的路由器
    capacity_factors = [1.0, 1.5, 2.0, 3.0]
    
    for cf in capacity_factors:
        router = LoadBalancedRouter(hidden_dim, num_experts, capacity_factor=cf)
        weights, experts, aux_loss = router(input_data, return_aux_loss=True)
        
        # 计算每个专家的实际负载
        expert_loads = []
        for i in range(num_experts):
            load = (experts == i).sum().item()
            expert_loads.append(load)
        
        print(f"\n容量因子: {cf}")
        print(f"  专家负载: {expert_loads}")
        print(f"  最大负载: {max(expert_loads)}")
        print(f"  负载均衡损失: {aux_loss.item():.4f}")


def demonstrate_expert_specialization():
    """演示专家专业化现象"""
    print("\n\n专家专业化演示")
    print("="*60)
    
    class SpecializedMoE(nn.Cell):
        """带有专业化专家的MoE层"""
        
        def __init__(self, input_dim, output_dim, num_experts):
            super().__init__()
            self.num_experts = num_experts
            self.router = SimpleRouter(input_dim, num_experts)
            
            # 创建专业化的专家
            self.experts = nn.CellList()
            for i in range(num_experts):
                # 每个专家有不同的激活函数，模拟专业化
                expert = nn.SequentialCell([
                    nn.Dense(input_dim, output_dim),
                    nn.ReLU() if i % 3 == 0 else (nn.Tanh() if i % 3 == 1 else nn.GELU()),
                    nn.Dense(output_dim, output_dim)
                ])
                self.experts.append(expert)
        
        def construct(self, x):
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = x.reshape(-1, hidden_dim)
            
            # 路由
            logits = self.router(x_flat)
            weights, selected = ops.topk(logits, k=2)
            weights = ops.softmax(weights, axis=-1)
            
            # 通过专家处理
            output = ops.zeros_like(x_flat)
            for i in range(self.num_experts):
                mask = (selected == i).any(axis=-1)
                if mask.any():
                    token_indices = ops.nonzero(mask).squeeze(-1)
                    expert_input = x_flat[token_indices]
                    expert_output = self.experts[i](expert_input)
                    
                    # 获取权重
                    expert_weights = ops.zeros(token_indices.shape[0])
                    for j, idx in enumerate(token_indices):
                        positions = ops.nonzero(selected[idx] == i).squeeze(-1)
                        if positions.numel() > 0:
                            expert_weights[j] = weights[idx, positions].sum()
                    
                    output[token_indices] += expert_weights.unsqueeze(-1) * expert_output
            
            return output.reshape(batch_size, seq_len, -1), selected.reshape(batch_size, seq_len, -1)
    
    # 创建模型
    model = SpecializedMoE(64, 64, 6)
    
    # 创建不同特征的输入
    test_inputs = {
        "High Frequency": ops.randn(2, 20, 64) * ops.sin(ops.arange(20).reshape(1, -1, 1) * 0.5),
        "Low Frequency": ops.randn(2, 20, 64) * ops.cos(ops.arange(20).reshape(1, -1, 1) * 0.1),
        "Sparse Features": ops.randn(2, 20, 64) * (ops.rand(2, 20, 64) > 0.7).astype(mindspore.float32),
        "Dense Features": ops.randn(2, 20, 64) + 1.0
    }
    
    print("\n不同输入特征的专家选择:")
    for feature_name, input_data in test_inputs.items():
        output, selected_experts = model(input_data)
        
        # 统计每个专家被选择的频率
        expert_freq = ops.zeros(model.num_experts)
        for i in range(model.num_experts):
            expert_freq[i] = (selected_experts == i).sum()
        
        print(f"\n{feature_name}:")
        print(f"  专家选择频率: {expert_freq.asnumpy()}")
        print(f"  主要专家: {expert_freq.argmax().item()}")


if __name__ == "__main__":
    # 运行所有演示
    demonstrate_routing_strategies()
    analyze_capacity_constraints()
    demonstrate_expert_specialization()
    
    print("\n\n演示完成！")
    print("查看生成的图片以了解路由模式的可视化结果。")
