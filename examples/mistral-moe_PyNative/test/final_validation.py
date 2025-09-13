# -*- coding: utf-8 -*-
"""
最终验证脚本 - 验证所有修复后的功能
"""

import time
import json
import mindspore
from mindspore import context, ops
import traceback

# 设置动态图模式
context.set_context(mode=context.PYNATIVE_MODE)

from models.mistral.configuration_mistral import MistralConfig, MoeConfig
from models.mistral.modeling_mistral import MistralForCausalLM, MistralMoELayer

class FinalValidator:
    def __init__(self):
        self.results = {}
        
    def test_basic_functionality(self):
        """测试基础功能"""
        print("测试1: 基础模型功能")
        print("-" * 40)
        
        try:
            # 标准模型
            config = MistralConfig(
                vocab_size=100,
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                intermediate_size=128
            )
            
            model = MistralForCausalLM(config)
            input_ids = ops.randint(0, config.vocab_size, (2, 8))
            
            # 前向传播
            outputs = model(input_ids)
            
            # 检查输出
            if isinstance(outputs, (list, tuple)):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs
                
            expected_shape = (2, 8, config.vocab_size)
            assert logits.shape == expected_shape, f"形状错误: {logits.shape} != {expected_shape}"
            
            # 数值检查
            assert not ops.isnan(logits).any(), "输出包含NaN"
            assert not ops.isinf(logits).any(), "输出包含Inf"
            
            print(f"✓ 标准模型测试通过 - 输出形状: {logits.shape}")
            self.results["基础功能"] = {"状态": "通过", "形状": list(logits.shape)}
            return True
            
        except Exception as e:
            print(f"✗ 基础功能测试失败: {e}")
            self.results["基础功能"] = {"状态": "失败", "错误": str(e)}
            return False
    
    def test_moe_functionality(self):
        """测试MoE功能"""
        print("\n测试2: MoE模型功能")
        print("-" * 40)
        
        try:
            # MoE模型
            config = MistralConfig(
                vocab_size=50,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                intermediate_size=64,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
            )
            
            model = MistralForCausalLM(config)
            input_ids = ops.randint(0, config.vocab_size, (1, 10))
            
            # 前向传播
            outputs = model(input_ids)
            
            if isinstance(outputs, (list, tuple)):
                logits = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                logits = outputs
                
            expected_shape = (1, 10, config.vocab_size)
            assert logits.shape == expected_shape, f"形状错误: {logits.shape} != {expected_shape}"
            
            # 数值检查
            assert not ops.isnan(logits).any(), "输出包含NaN"
            assert not ops.isinf(logits).any(), "输出包含Inf"
            
            print(f"✓ MoE模型测试通过 - 输出形状: {logits.shape}")
            self.results["MoE功能"] = {"状态": "通过", "形状": list(logits.shape)}
            return True
            
        except Exception as e:
            print(f"✗ MoE功能测试失败: {e}")
            self.results["MoE功能"] = {"状态": "失败", "错误": str(e)}
            return False
    
    def test_moe_routing(self):
        """测试MoE路由机制"""
        print("\n测试3: MoE路由机制")
        print("-" * 40)
        
        try:
            config = MistralConfig(
                hidden_size=32,
                intermediate_size=64,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
            )
            
            moe_layer = MistralMoELayer(config)
            input_tensor = ops.randn(2, 8, config.hidden_size)
            
            # 路由测试
            output = moe_layer(input_tensor)
            
            assert output.shape == input_tensor.shape, f"形状错误: {output.shape} != {input_tensor.shape}"
            assert not ops.isnan(output).any(), "输出包含NaN"
            assert not ops.isinf(output).any(), "输出包含Inf"
            
            # 测试路由分布
            hidden_flat = input_tensor.reshape(-1, config.hidden_size)
            router_logits = moe_layer.gate(hidden_flat)
            _, selected_experts = ops.topk(router_logits, config.moe.num_experts_per_tok)
            
            # 检查专家选择
            assert (selected_experts >= 0).all(), "专家索引包含负数"
            assert (selected_experts < config.moe.num_experts).all(), "专家索引超出范围"
            
            print(f"✓ MoE路由测试通过 - 输出形状: {output.shape}")
            self.results["MoE路由"] = {"状态": "通过", "专家数": config.moe.num_experts}
            return True
            
        except Exception as e:
            print(f"✗ MoE路由测试失败: {e}")
            self.results["MoE路由"] = {"状态": "失败", "错误": str(e)}
            return False
    
    def test_generation(self):
        """测试文本生成"""
        print("\n测试4: 文本生成功能")
        print("-" * 40)
        
        try:
            config = MistralConfig(
                vocab_size=20,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                intermediate_size=32,
                moe=MoeConfig(num_experts=2, num_experts_per_tok=1)
            )
            
            model = MistralForCausalLM(config)
            model.set_train(False)
            
            # 生成测试
            prompt = ops.randint(1, config.vocab_size-1, (1, 3))
            generated = prompt
            
            for i in range(5):
                outputs = model(generated)
                if isinstance(outputs, (list, tuple)):
                    logits = outputs[1] if len(outputs) > 1 else outputs[0]
                else:
                    logits = outputs
                    
                next_token = ops.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
                generated = ops.concat([generated, next_token], axis=1)
            
            assert generated.shape[1] == 8, f"生成长度错误: {generated.shape[1]} != 8"
            
            print(f"✓ 文本生成测试通过 - 最终长度: {generated.shape[1]}")
            print(f"  生成序列: {generated.asnumpy().flatten()}")
            self.results["文本生成"] = {"状态": "通过", "长度": generated.shape[1]}
            return True
            
        except Exception as e:
            print(f"✗ 文本生成测试失败: {e}")
            self.results["文本生成"] = {"状态": "失败", "错误": str(e)}
            return False
    
    def test_visualization(self):
        """测试可视化功能"""
        print("\n测试5: 可视化功能")
        print("-" * 40)
        
        try:
            from course.code_examples.moe_routing_demo import SimpleRouter, visualize_routing_patterns
            
            router = SimpleRouter(32, 4)
            input_data = ops.randn(1, 8, 32)
            
            # 可视化测试（不显示，只保存）
            print("开始可视化测试...")
            visualize_routing_patterns(router, input_data, "Test Routing Visualization")
            
            print("✓ 可视化测试通过")
            self.results["可视化"] = {"状态": "通过"}
            return True
            
        except Exception as e:
            print(f"✗ 可视化测试失败: {e}")
            self.results["可视化"] = {"状态": "失败", "错误": str(e)}
            return False
    
    def test_performance(self):
        """测试性能"""
        print("\n测试6: 性能基准")
        print("-" * 40)
        
        try:
            config = MistralConfig(
                vocab_size=100,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                intermediate_size=256,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)
            )
            
            model = MistralForCausalLM(config)
            model.set_train(False)
            
            # 预热
            warmup_input = ops.randint(0, config.vocab_size, (1, 10))
            for _ in range(3):
                _ = model(warmup_input)
            
            # 性能测试
            test_input = ops.randint(0, config.vocab_size, (2, 20))
            
            times = []
            for _ in range(5):
                start = time.time()
                _ = model(test_input)
                times.append(time.time() - start)
            
            avg_time = sum(times[1:]) / len(times[1:])  # 排除第一次
            throughput = (2 * 20) / avg_time  # tokens/s
            
            print(f"✓ 性能测试通过")
            print(f"  平均时间: {avg_time*1000:.2f}ms")
            print(f"  吞吐量: {throughput:.1f} tokens/s")
            
            self.results["性能"] = {
                "状态": "通过", 
                "平均时间": f"{avg_time*1000:.2f}ms",
                "吞吐量": f"{throughput:.1f} tokens/s"
            }
            return True
            
        except Exception as e:
            print(f"✗ 性能测试失败: {e}")
            self.results["性能"] = {"状态": "失败", "错误": str(e)}
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("最终验证测试套件")
        print("=" * 60)
        
        tests = [
            self.test_basic_functionality,
            self.test_moe_functionality,
            self.test_moe_routing,
            self.test_generation,
            self.test_visualization,
            self.test_performance
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
        
        # 生成报告
        print("\n" + "=" * 60)
        print("最终测试报告")
        print("=" * 60)
        print(f"通过: {passed}/{total} ({passed/total*100:.1f}%)")
        
        report = {
            "总测试数": total,
            "通过数": passed,
            "失败数": total - passed,
            "成功率": f"{passed/total*100:.1f}%",
            "详细结果": self.results
        }
        
        with open('final_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        if passed == total:
            print("🎉 所有测试都通过了！")
            print("✅ 模型迁移和修复完全成功！")
        else:
            print(f"⚠️ {total - passed} 个测试失败")
        
        print(f"\n详细报告已保存至: final_validation_report.json")
        return passed == total

if __name__ == "__main__":
    validator = FinalValidator()
    success = validator.run_all_tests()
