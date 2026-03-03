"""
QLoRA 训练示例

演示如何使用 QLoRA 进行大模型高效微调。
"""

import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor, context, nn
    from mindspore.train import Model
    from mindspore.nn import Adam, SoftmaxCrossEntropyWithLogits
    
    # 设置运行模式
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    
    # 导入量化模块
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from msbnb import (
        Linear4bitWithLoRA,
        Linear8bitWithLoRA,
        LoRALinear,
        freeze_model_except_lora,
        print_lora_info,
        get_model_size,
        compare_model_sizes
    )
    
    print("=" * 60)
    print("QLoRA 训练示例")
    print("=" * 60)
    
    # ========== 定义示例模型 ==========
    class SimpleLanguageModel(nn.Cell):
        """简单的语言模型"""
        def __init__(self, vocab_size=10000, hidden_size=768, num_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            
            self.layers = nn.CellList([
                nn.SequentialCell([
                    nn.Dense(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dense(hidden_size * 4, hidden_size),
                ])
                for _ in range(num_layers)
            ])
            
            self.lm_head = nn.Dense(hidden_size, vocab_size)
        
        def construct(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = x + layer(x)
            logits = self.lm_head(x)
            return logits
    
    # ========== 示例 1: 创建 QLoRA 模型 ==========
    print("\n[1] 创建 QLoRA 模型")
    print("-" * 60)
    
    # 创建原始模型
    model_fp = SimpleLanguageModel(vocab_size=10000, hidden_size=768, num_layers=2)
    print("原始模型创建完成")
    
    # 获取原始模型大小
    size_fp = get_model_size(model_fp)
    print(f"原始模型大小: {size_fp['total_size_mb']:.2f} MB")
    print(f"总参数量: {size_fp['total_params']:,}")
    
    # 转换为 QLoRA 模型
    print("\n转换为 QLoRA 模型...")
    model_qlora = SimpleLanguageModel(vocab_size=10000, hidden_size=768, num_layers=2)
    
    # 替换中间层为 QLoRA 层
    for i, layer_seq in enumerate(model_qlora.layers):
        # 获取原始层
        fc1 = layer_seq[0]  # Dense(768, 3072)
        fc2 = layer_seq[2]  # Dense(3072, 768)
        
        # 转换为 QLoRA 层
        qlora_fc1 = Linear4bitWithLoRA.from_linear(
            fc1,
            group_size=128,
            compress_statistics=True,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        qlora_fc2 = Linear4bitWithLoRA.from_linear(
            fc2,
            group_size=128,
            compress_statistics=True,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        
        # 替换层
        layer_seq[0] = qlora_fc1
        layer_seq[2] = qlora_fc2
        
        print(f"\n已转换第 {i+1} 层")
    
    # 冻结非 LoRA 参数
    print("\n冻结非 LoRA 参数...")
    frozen_count, trainable_count = freeze_model_except_lora(model_qlora)
    print(f"冻结参数: {frozen_count}")
    print(f"可训练参数: {trainable_count}")
    
    # 打印 LoRA 信息
    print("\n")
    print_lora_info(model_qlora)
    
    # 比较模型大小
    print("\n模型大小对比:")
    size_qlora = get_model_size(model_qlora)
    print(f"FP32 模型: {size_fp['total_size_mb']:.2f} MB")
    print(f"QLoRA 模型: {size_qlora['total_size_mb']:.2f} MB")
    print(f"显存节省: {(1 - size_qlora['total_size_mb']/size_fp['total_size_mb'])*100:.1f}%")
    
    # ========== 示例 2: QLoRA 训练 ==========
    print("\n[2] QLoRA 训练")
    print("-" * 60)
    
    # 准备训练数据（模拟）
    batch_size = 8
    seq_length = 32
    
    # 创建模拟数据
    input_ids = Tensor(np.random.randint(0, 10000, (batch_size, seq_length)), dtype=ms.int32)
    labels = Tensor(np.random.randint(0, 10000, (batch_size, seq_length)), dtype=ms.int32)
    
    print(f"训练数据形状: {input_ids.shape}")
    
    # 定义损失函数和优化器
    loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    # 只优化 LoRA 参数
    lora_params = [param for name, param in model_qlora.parameters_and_names() 
                   if param.requires_grad]
    optimizer = Adam(lora_params, learning_rate=1e-4)
    
    print(f"优化器参数数量: {len(lora_params)}")
    
    # 定义训练步骤
    def forward_fn(input_ids, labels):
        logits = model_qlora(input_ids)
        # 重塑 logits 和 labels
        logits = logits.reshape(-1, 10000)
        labels = labels.reshape(-1)
        loss = loss_fn(logits, labels)
        return loss
    
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)
    
    # 训练循环
    print("\n开始训练...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        # 前向传播和反向传播
        loss, grads = grad_fn(input_ids, labels)
        
        # 更新参数
        optimizer(grads)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.asnumpy():.4f}")
    
    print("\n训练完成！")
    
    # ========== 示例 3: 单独使用 LoRA 层 ==========
    print("\n[3] 单独使用 LoRA 层")
    print("-" * 60)
    
    # 创建 LoRA 层
    lora_layer = LoRALinear(
        in_features=768,
        out_features=3072,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    print(f"LoRA 层创建完成")
    print(f"  输入维度: {lora_layer.in_features}")
    print(f"  输出维度: {lora_layer.out_features}")
    print(f"  秩: {lora_layer.r}")
    print(f"  缩放因子: {lora_layer.scaling:.2f}")
    
    # 前向传播
    x = Tensor(np.random.randn(32, 768).astype(np.float32))
    out = lora_layer(x)
    print(f"\n前向传播:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    
    # 获取合并后的权重
    delta_W = lora_layer.get_merged_weight()
    print(f"\n权重增量形状: {delta_W.shape}")
    print(f"权重增量范围: [{delta_W.min():.6f}, {delta_W.max():.6f}]")
    
    # ========== 示例 4: INT8 + LoRA ==========
    print("\n[4] INT8 + LoRA")
    print("-" * 60)
    
    # 创建 INT8 + LoRA 层
    layer_int8_lora = Linear8bitWithLoRA(
        in_features=768,
        out_features=3072,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    print("INT8 + LoRA 层创建完成")
    
    # 前向传播
    x = Tensor(np.random.randn(32, 768).astype(np.float32))
    out = layer_int8_lora(x)
    print(f"输出形状: {out.shape}")
    
    # 打印可训练参数
    trainable_params = [(name, param) for name, param in layer_int8_lora.parameters_and_names() 
                        if param.requires_grad]
    print(f"\n可训练参数数量: {len(trainable_params)}")
    for name, param in trainable_params:
        print(f"  {name}: {param.shape}")
    
    # ========== 示例 5: 参数效率对比 ==========
    print("\n[5] 参数效率对比")
    print("-" * 60)
    
    # 计算不同方法的参数量
    hidden_size = 768
    intermediate_size = 3072
    
    # 全量微调
    full_params = hidden_size * intermediate_size + intermediate_size * hidden_size
    
    # LoRA (r=8)
    r = 8
    lora_params = (hidden_size * r + r * intermediate_size) * 2  # 两个方向
    
    # LoRA (r=16)
    r_16 = 16
    lora_params_16 = (hidden_size * r_16 + r_16 * intermediate_size) * 2
    
    print(f"{'方法':<20} {'参数量':<15} {'相对比例':<10}")
    print("-" * 50)
    print(f"{'全量微调':<20} {full_params:>12,}   {100.0:>8.2f}%")
    print(f"{'LoRA (r=8)':<20} {lora_params:>12,}   {lora_params/full_params*100:>8.2f}%")
    print(f"{'LoRA (r=16)':<20} {lora_params_16:>12,}   {lora_params_16/full_params*100:>8.2f}%")
    
    print("\n显存节省:")
    print(f"  LoRA (r=8):  节省 {(1 - lora_params/full_params)*100:.1f}%")
    print(f"  LoRA (r=16): 节省 {(1 - lora_params_16/full_params)*100:.1f}%")
    
    # ========== 示例 6: 推荐配置 ==========
    print("\n[6] 推荐配置")
    print("-" * 60)
    
    print("根据模型大小选择 LoRA 配置:")
    print()
    print("小模型 (< 1B 参数):")
    print("  - r: 8-16")
    print("  - lora_alpha: 16-32")
    print("  - lora_dropout: 0.05-0.1")
    print()
    print("中等模型 (1B-10B 参数):")
    print("  - r: 16-32")
    print("  - lora_alpha: 32-64")
    print("  - lora_dropout: 0.1")
    print()
    print("大模型 (> 10B 参数):")
    print("  - r: 32-64")
    print("  - lora_alpha: 64-128")
    print("  - lora_dropout: 0.1")
    print()
    print("量化配置:")
    print("  - INT4: group_size=128, compress_statistics=True")
    print("  - INT8: per_channel=True, symmetric=True")
    
    print("示例运行完成！")

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装 MindSpore 并正确设置路径")
except Exception as e:
    print(f"运行错误: {e}")
    import traceback
    traceback.print_exc()
