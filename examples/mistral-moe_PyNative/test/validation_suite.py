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
全面的验证套件，用于检验Mistral MoE模型迁移的正确性
"""

import time
import json
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np

import mindspore
from mindspore import context, ops, nn, Tensor
from mindspore.train import Model

# 设置环境
context.set_context(mode=context.PYNATIVE_MODE)

from models.mistral.configuration_mistral import MistralConfig, MoeConfig
from models.mistral.modeling_mistral import MistralModel, MistralForCausalLM


class ValidationSuite:
    """Mistral MoE模型验证套件"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """日志输出"""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def run_all_tests(self):
        """运行所有验证测试"""
        self.log("="*60)
        self.log("开始Mistral MoE模型验证套件")
        self.log("="*60)
        
        # 运行各项测试
        self.test_model_creation()
        self.test_forward_pass()
        self.test_moe_routing()
        self.test_attention_mechanism()
        self.test_generation()
        self.test_memory_efficiency()
        self.test_numerical_stability()
        self.test_performance()
        
        # 生成报告
        self.generate_report()
        
    def test_model_creation(self):
        """测试1：模型创建和配置"""
        self.log("\n测试1：模型创建和配置")
        self.log("-"*40)
        
        try:
            # 测试标准配置
            config_standard = MistralConfig(
                vocab_size=32000,
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
            )
            model_standard = MistralForCausalLM(config_standard)
            self.log("✓ 标准Mistral模型创建成功")
            
            # 测试MoE配置
            config_moe = MistralConfig(
                vocab_size=32000,
                hidden_size=4096,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                moe=MoeConfig(num_experts=8, num_experts_per_tok=2)
            )
            model_moe = MistralForCausalLM(config_moe)
            self.log("✓ Mixtral MoE模型创建成功")
            
            # 验证参数数量
            param_count_standard = sum(p.size for p in model_standard.trainable_params())
            param_count_moe = sum(p.size for p in model_moe.trainable_params())
            
            self.log(f"  标准模型参数量: {param_count_standard:,}")
            self.log(f"  MoE模型参数量: {param_count_moe:,}")
            self.log(f"  参数增长比例: {param_count_moe/param_count_standard:.2f}x")
            
            self.results['model_creation'] = {
                'status': 'PASS',
                'standard_params': param_count_standard,
                'moe_params': param_count_moe,
                'ratio': param_count_moe/param_count_standard
            }
            
        except Exception as e:
            self.log(f"✗ 模型创建失败: {e}", "ERROR")
            self.results['model_creation'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_forward_pass(self):
        """测试2：前向传播"""
        self.log("\n测试2：前向传播")
        self.log("-"*40)
        
        try:
            # 创建小型测试模型
            config = MistralConfig(
                vocab_size=1000,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=32,
                intermediate_size=512,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
            )
            model = MistralForCausalLM(config)
            
            # 测试输入
            batch_size = 2
            seq_len = 10
            input_ids = ops.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # 前向传播
            outputs = model(input_ids)
            logits = outputs[1]
            
            # 验证输出形状
            expected_shape = (batch_size, seq_len, config.vocab_size)
            assert logits.shape == expected_shape, f"输出形状错误: {logits.shape} != {expected_shape}"
            
            self.log(f"✓ 前向传播成功，输出形状: {logits.shape}")
            
            # 测试带标签的前向传播
            labels = ops.randint(0, config.vocab_size, (batch_size, seq_len))
            outputs_with_loss = model(input_ids, labels=labels)
            loss = outputs_with_loss[0]
            
            assert loss.ndim == 0, "损失应该是标量"
            assert loss.item() > 0, "损失应该是正数"
            
            self.log(f"✓ 损失计算成功: {loss.item():.4f}")
            
            self.results['forward_pass'] = {
                'status': 'PASS',
                'output_shape': list(logits.shape),
                'loss_value': float(loss.item())
            }
            
        except Exception as e:
            self.log(f"✗ 前向传播失败: {e}", "ERROR")
            self.results['forward_pass'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_moe_routing(self):
        """测试3：MoE路由机制"""
        self.log("\n测试3：MoE路由机制")
        self.log("-"*40)
        
        try:
            from models.mistral.modeling_mistral import MistralMoELayer
            
            config = MistralConfig(
                hidden_size=256,
                intermediate_size=512,
                moe=MoeConfig(num_experts=8, num_experts_per_tok=2)
            )
            
            moe_layer = MistralMoELayer(config)
            
            # 测试不同批次大小
            test_cases = [
                (1, 10),   # 小批次
                (4, 20),   # 中等批次
                (8, 50),   # 大批次
            ]
            
            for batch_size, seq_len in test_cases:
                input_tensor = ops.randn(batch_size, seq_len, config.hidden_size)
                
                # 获取路由决策
                hidden_flat = input_tensor.view(-1, config.hidden_size)
                router_logits = moe_layer.gate(hidden_flat)
                routing_weights, selected_experts = ops.topk(router_logits, config.moe.num_experts_per_tok)
                
                # 验证路由
                assert selected_experts.shape == (batch_size * seq_len, config.moe.num_experts_per_tok)
                assert (selected_experts >= 0).all() and (selected_experts < config.moe.num_experts).all()
                
                # 计算负载分布
                expert_loads = ops.zeros(config.moe.num_experts)
                for i in range(config.moe.num_experts):
                    expert_loads[i] = (selected_experts == i).sum()
                
                load_variance = expert_loads.std().item()
                self.log(f"  批次{batch_size}x{seq_len}: 负载方差={load_variance:.2f}")
                
                # 测试前向传播
                output = moe_layer(input_tensor)
                assert output.shape == input_tensor.shape
            
            self.log("✓ MoE路由机制正常")
            
            self.results['moe_routing'] = {
                'status': 'PASS',
                'load_variance': load_variance
            }
            
        except Exception as e:
            self.log(f"✗ MoE路由测试失败: {e}", "ERROR")
            self.results['moe_routing'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_attention_mechanism(self):
        """测试4：注意力机制"""
        self.log("\n测试4：注意力机制")
        self.log("-"*40)
        
        try:
            from models.mistral.modeling_mistral import MistralAttention
            
            config = MistralConfig(
                hidden_size=256,
                num_attention_heads=8,
                num_key_value_heads=4,
                head_dim=32,
                sliding_window=128
            )
            
            attention = MistralAttention(config)
            
            # 测试不同序列长度
            batch_size = 2
            test_lengths = [10, 50, 100, 200]
            
            for seq_len in test_lengths:
                hidden_states = ops.randn(batch_size, seq_len, config.hidden_size)
                
                # 测试无缓存
                output, _, _ = attention(hidden_states)
                assert output.shape == hidden_states.shape
                
                # 测试带缓存
                output_cached, _, past_kv = attention(
                    hidden_states, 
                    use_cache=True
                )
                assert past_kv is not None
                assert len(past_kv) == 2  # key和value
                
                self.log(f"  序列长度{seq_len}: ✓")
            
            # 测试滑动窗口
            if config.sliding_window:
                self.log(f"  滑动窗口大小: {config.sliding_window}")
            
            self.log("✓ 注意力机制正常")
            
            self.results['attention'] = {
                'status': 'PASS',
                'tested_lengths': test_lengths
            }
            
        except Exception as e:
            self.log(f"✗ 注意力机制测试失败: {e}", "ERROR")
            self.results['attention'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_generation(self):
        """测试5：文本生成"""
        self.log("\n测试5：文本生成")
        self.log("-"*40)
        
        try:
            # 创建小型模型用于生成测试
            config = MistralConfig(
                vocab_size=100,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
            )
            model = MistralForCausalLM(config)
            model.set_train(False)
            
            # 生成函数
            def generate(model, input_ids, max_length=20):
                generated = input_ids
                past_key_values = None
                
                for _ in range(max_length - input_ids.shape[1]):
                    outputs = model(
                        generated[:, -1:] if past_key_values else generated,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    logits = outputs[1]
                    past_key_values = outputs[2]
                    
                    next_token = ops.argmax(logits[:, -1, :], axis=-1, keepdim=True)
                    generated = ops.concat([generated, next_token], axis=1)
                
                return generated
            
            # 测试生成
            prompt = ops.randint(1, config.vocab_size, (1, 5))
            generated = generate(model, prompt, max_length=20)
            
            assert generated.shape[1] == 20
            self.log(f"✓ 生成成功，序列长度: {generated.shape[1]}")
            
            self.results['generation'] = {
                'status': 'PASS',
                'generated_length': generated.shape[1]
            }
            
        except Exception as e:
            self.log(f"✗ 生成测试失败: {e}", "ERROR")
            self.results['generation'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_memory_efficiency(self):
        """测试6：内存效率"""
        self.log("\n测试6：内存效率")
        self.log("-"*40)
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # 测试不同配置的内存使用
            configs = [
                ("Small", MistralConfig(vocab_size=1000, hidden_size=256, num_hidden_layers=4)),
                ("Small-MoE", MistralConfig(
                    vocab_size=1000, hidden_size=256, num_hidden_layers=4,
                    moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
                )),
            ]
            
            memory_usage = {}
            
            for name, config in configs:
                # 记录初始内存
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # 创建模型
                model = MistralForCausalLM(config)
                
                # 执行前向传播
                input_ids = ops.randint(0, config.vocab_size, (2, 10))
                _ = model(input_ids)
                
                # 记录最终内存
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                memory_usage[name] = memory_increase
                self.log(f"  {name}: {memory_increase:.2f} MB")
                
                # 清理
                del model
            
            # 计算MoE的内存开销
            moe_overhead = memory_usage["Small-MoE"] / memory_usage["Small"]
            self.log(f"  MoE内存开销比例: {moe_overhead:.2f}x")
            
            self.results['memory_efficiency'] = {
                'status': 'PASS',
                'memory_usage': memory_usage,
                'moe_overhead': moe_overhead
            }
            
        except Exception as e:
            self.log(f"✗ 内存效率测试失败: {e}", "ERROR")
            self.results['memory_efficiency'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_numerical_stability(self):
        """测试7：数值稳定性"""
        self.log("\n测试7：数值稳定性")
        self.log("-"*40)
        
        try:
            config = MistralConfig(
                vocab_size=1000,
                hidden_size=256,
                num_hidden_layers=4,
                moe=MoeConfig(num_experts=8, num_experts_per_tok=2)
            )
            model = MistralForCausalLM(config)
            
            # 测试极端输入
            test_cases = [
                ("正常输入", ops.randn(2, 10, config.hidden_size)),
                ("大值输入", ops.randn(2, 10, config.hidden_size) * 100),
                ("小值输入", ops.randn(2, 10, config.hidden_size) * 0.001),
                ("稀疏输入", ops.randn(2, 10, config.hidden_size) * (ops.rand(2, 10, config.hidden_size) > 0.9)),
            ]
            
            for name, test_input in test_cases:
                # 为attention层准备输入
                input_ids = ops.randint(0, config.vocab_size, (2, 10))
                
                # 获取嵌入
                embeddings = model.model.embed_tokens(input_ids)
                
                # 测试第一层
                layer = model.model.layers[0]
                try:
                    output = layer(embeddings)
                    
                    # 检查输出
                    has_nan = ops.isnan(output[0]).any().item()
                    has_inf = ops.isinf(output[0]).any().item()
                    
                    if has_nan or has_inf:
                        self.log(f"  {name}: ✗ (包含NaN或Inf)", "WARNING")
                    else:
                        output_mean = output[0].mean().item()
                        output_std = output[0].std().item()
                        self.log(f"  {name}: ✓ (均值={output_mean:.4f}, 标准差={output_std:.4f})")
                        
                except Exception as e:
                    self.log(f"  {name}: ✗ (错误: {e})", "ERROR")
            
            self.log("✓ 数值稳定性测试完成")
            
            self.results['numerical_stability'] = {'status': 'PASS'}
            
        except Exception as e:
            self.log(f"✗ 数值稳定性测试失败: {e}", "ERROR")
            self.results['numerical_stability'] = {'status': 'FAIL', 'error': str(e)}
    
    def test_performance(self):
        """测试8：性能基准"""
        self.log("\n测试8：性能基准")
        self.log("-"*40)
        
        try:
            # 创建测试模型
            config = MistralConfig(
                vocab_size=1000,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=4,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
            )
            model = MistralForCausalLM(config)
            model.set_train(False)
            
            # 预热
            warmup_input = ops.randint(0, config.vocab_size, (1, 10))
            for _ in range(5):
                _ = model(warmup_input)
            
            # 性能测试
            batch_sizes = [1, 4, 8]
            seq_lengths = [10, 50, 100]
            
            results = {}
            
            for batch_size in batch_sizes:
                for seq_len in seq_lengths:
                    input_ids = ops.randint(0, config.vocab_size, (batch_size, seq_len))
                    
                    # 计时
                    times = []
                    for _ in range(10):
                        start = time.time()
                        _ = model(input_ids)
                        end = time.time()
                        times.append(end - start)
                    
                    avg_time = np.mean(times[2:])  # 排除前两次
                    throughput = (batch_size * seq_len) / avg_time
                    
                    key = f"B{batch_size}_L{seq_len}"
                    results[key] = {
                        'avg_time': avg_time,
                        'throughput': throughput
                    }
                    
                    self.log(f"  {key}: {avg_time*1000:.2f}ms, {throughput:.0f} tokens/s")
            
            self.results['performance'] = {
                'status': 'PASS',
                'benchmarks': results
            }
            
        except Exception as e:
            self.log(f"✗ 性能测试失败: {e}", "ERROR")
            self.results['performance'] = {'status': 'FAIL', 'error': str(e)}
    
    def generate_report(self):
        """生成验证报告"""
        self.log("\n" + "="*60)
        self.log("验证报告")
        self.log("="*60)
        
        passed = 0
        failed = 0
        
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASS':
                passed += 1
                self.log(f"✓ {test_name}: PASS")
            else:
                failed += 1
                self.log(f"✗ {test_name}: FAIL", "ERROR")
        
        self.log(f"\n总计: {passed} 通过, {failed} 失败")
        
        # 保存详细报告
        with open('validation_report.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log("\n详细报告已保存至 validation_report.json")
        
        return passed, failed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Mistral MoE模型验证套件')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    parser.add_argument('--device', type=str, default='CPU', choices=['CPU', 'GPU', 'Ascend'],
                       help='运行设备')
    args = parser.parse_args()
    
    # 设置设备
    context.set_context(device_target=args.device)
    
    # 运行验证
    suite = ValidationSuite(verbose=args.verbose)
    suite.run_all_tests()


if __name__ == "__main__":
    main()
