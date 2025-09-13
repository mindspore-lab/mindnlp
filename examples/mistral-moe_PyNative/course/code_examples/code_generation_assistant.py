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
代码生成助手 - 基于Mistral MoE模型

本应用案例展示了如何使用Mistral MoE模型进行智能代码生成，
包括：
1. 多语言代码生成（Python、JavaScript、Java等）
2. 代码补全和修复
3. 代码注释生成
4. 代码质量分析
5. 专家路由优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mindspore
from mindspore import nn, ops, context, Tensor
import numpy as np
import time
import json
import re
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# 导入项目模型
from models.mistral.configuration_mistral import MistralConfig, MoeConfig
from models.mistral.modeling_mistral import MistralModel
from models.mistral.tokenization_mistral import MistralTokenizer

# 设置动态图模式
context.set_context(mode=context.PYNATIVE_MODE)


class CodeGenerationAssistant:
    """代码生成助手"""
    
    def __init__(self, model_path: str = None, max_length: int = 2048):
        """
        初始化代码生成助手
        
        Args:
            model_path: 模型路径，如果为None则使用默认配置
            max_length: 最大序列长度
        """
        self.max_length = max_length
        
        # 初始化配置
        if model_path and os.path.exists(model_path):
            self.config = MistralConfig.from_pretrained(model_path)
        else:
            # 使用小型配置用于演示
            self.config = MistralConfig(
                vocab_size=32000,
                hidden_size=512,
                intermediate_size=1024,
                num_hidden_layers=6,
                num_attention_heads=8,
                num_key_value_heads=4,
                hidden_act="silu",
                max_position_embeddings=4096,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                pad_token_id=None,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
                rope_theta=10000.0,
                sliding_window=4096,
                attention_dropout=0.0,
                moe=MoeConfig(num_experts=4, num_experts_per_tok=2)  # MoE配置
            )
        
        # 初始化模型
        self.model = MistralModel(self.config)
        
        # 初始化分词器
        self.tokenizer = self._create_code_tokenizer()
        
        # 代码生成提示模板
        self.code_prompts = {
            "python": {
                "function": "请用Python编写一个函数，实现以下功能：\n\n",
                "class": "请用Python编写一个类，实现以下功能：\n\n",
                "script": "请用Python编写一个脚本，实现以下功能：\n\n",
                "complete": "请补全以下Python代码：\n\n",
                "comment": "请为以下Python代码添加详细的中文注释：\n\n"
            },
            "javascript": {
                "function": "请用JavaScript编写一个函数，实现以下功能：\n\n",
                "class": "请用JavaScript编写一个类，实现以下功能：\n\n",
                "script": "请用JavaScript编写一个脚本，实现以下功能：\n\n",
                "complete": "请补全以下JavaScript代码：\n\n",
                "comment": "请为以下JavaScript代码添加详细的中文注释：\n\n"
            },
            "java": {
                "function": "请用Java编写一个方法，实现以下功能：\n\n",
                "class": "请用Java编写一个类，实现以下功能：\n\n",
                "script": "请用Java编写一个程序，实现以下功能：\n\n",
                "complete": "请补全以下Java代码：\n\n",
                "comment": "请为以下Java代码添加详细的中文注释：\n\n"
            }
        }
        
        # 代码质量检查规则
        self.code_quality_rules = {
            "python": {
                "indentation": r"^(\s{4})+",  # 4空格缩进
                "naming": r"^[a-z_][a-z0-9_]*$",  # 小写下划线命名
                "docstring": r'""".*"""',  # 文档字符串
                "imports": r"^import\s+|^from\s+",  # 导入语句
            },
            "javascript": {
                "indentation": r"^(\s{2})+",  # 2空格缩进
                "naming": r"^[a-z][a-zA-Z0-9]*$",  # 驼峰命名
                "comments": r"//.*|/\*.*\*/",  # 注释
                "imports": r"^import\s+|^const\s+|^let\s+|^var\s+",  # 导入和声明
            },
            "java": {
                "indentation": r"^(\s{4})+",  # 4空格缩进
                "naming": r"^[A-Z][a-zA-Z0-9]*$",  # 驼峰命名
                "comments": r"//.*|/\*.*\*/",  # 注释
                "imports": r"^import\s+|^public\s+class\s+",  # 导入和类声明
            }
        }
        
        print(f"✅ 代码生成助手初始化完成")
        print(f"   - 模型配置: {self.config.hidden_size}维, {self.config.num_hidden_layers}层")
        print(f"   - MoE专家: {self.config.moe.num_experts}个专家, 每token使用{self.config.moe.num_experts_per_tok}个")
        print(f"   - 支持语言: Python, JavaScript, Java")
        print(f"   - 最大长度: {self.max_length}")
    
    def _create_code_tokenizer(self):
        """创建代码专用分词器"""
        class CodeTokenizer:
            def __init__(self):
                self.vocab_size = 32000
                self.pad_token_id = 0
                self.bos_token_id = 1
                self.eos_token_id = 2
                self.unk_token_id = 3
                
                # 创建代码词汇表
                self.char_to_id = {chr(i): i + 4 for i in range(32, 127)}  # 可打印ASCII字符
                self.char_to_id.update({
                    '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
                    '\n': 10, '\t': 9, ' ': 32  # 特殊字符
                })
                self.id_to_char = {v: k for k, v in self.char_to_id.items()}
            
            def encode(self, text: str, max_length: int = None) -> List[int]:
                """编码文本为token ID"""
                tokens = [self.bos_token_id]
                
                for char in text:
                    if char in self.char_to_id:
                        tokens.append(self.char_to_id[char])
                    else:
                        tokens.append(self.unk_token_id)
                
                tokens.append(self.eos_token_id)
                
                if max_length:
                    tokens = tokens[:max_length]
                    if len(tokens) < max_length:
                        tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
                
                return tokens
            
            def decode(self, token_ids: List[int]) -> str:
                """解码token ID为文本"""
                text = ""
                for token_id in token_ids:
                    if token_id in self.id_to_char:
                        char = self.id_to_char[token_id]
                        if char not in ['<pad>', '<bos>', '<eos>', '<unk>']:
                            text += char
                return text
        
        return CodeTokenizer()
    
    def _analyze_code_expert_usage(self, input_ids: Tensor, language: str) -> Dict:
        """分析代码生成中的专家使用情况"""
        try:
            # 根据编程语言调整专家分布（模拟）
            if language == "python":
                # Python代码可能更倾向于使用某些专家
                expert_dist = np.array([0.3, 0.25, 0.25, 0.2])
            elif language == "javascript":
                # JavaScript代码的专家分布
                expert_dist = np.array([0.25, 0.3, 0.2, 0.25])
            elif language == "java":
                # Java代码的专家分布
                expert_dist = np.array([0.2, 0.25, 0.3, 0.25])
            else:
                expert_dist = np.random.dirichlet(np.ones(self.config.moe.num_experts))
            
            expert_usage = {
                'total_tokens': input_ids.shape[1],
                'language': language,
                'expert_distribution': expert_dist.tolist(),
                'load_balance_score': np.random.uniform(0.75, 0.95),
                'specialization_score': np.random.uniform(0.7, 0.9),
                'code_complexity': self._analyze_code_complexity(input_ids)
            }
            
            return expert_usage
            
        except Exception as e:
            print(f"⚠️ 代码专家使用分析出错: {e}")
            return {
                'total_tokens': input_ids.shape[1],
                'language': language,
                'expert_distribution': [0.25] * self.config.moe.num_experts,
                'load_balance_score': 0.8,
                'specialization_score': 0.7,
                'code_complexity': 'medium'
            }
    
    def _analyze_code_complexity(self, input_ids: Tensor) -> str:
        """分析代码复杂度"""
        try:
            # 将token ID转换回文本
            text = self.tokenizer.decode(input_ids[0].asnumpy().tolist())
            
            # 简单的复杂度分析
            lines = text.split('\n')
            avg_line_length = np.mean([len(line) for line in lines if line.strip()])
            
            if avg_line_length > 80:
                return 'high'
            elif avg_line_length > 50:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'
    
    def _evaluate_code_quality(self, code: str, language: str) -> Dict:
        """评估代码质量"""
        quality_metrics = {
            'language': language,
            'total_lines': len(code.split('\n')),
            'code_length': len(code),
            'indentation_score': 0.0,
            'naming_score': 0.0,
            'comment_score': 0.0,
            'structure_score': 0.0,
            'overall_score': 0.0
        }
        
        try:
            lines = code.split('\n')
            rules = self.code_quality_rules.get(language, {})
            
            # 检查缩进
            indentation_matches = 0
            for line in lines:
                if line.strip() and re.match(rules.get('indentation', r'^\s*'), line):
                    indentation_matches += 1
            quality_metrics['indentation_score'] = indentation_matches / len(lines) if lines else 0
            
            # 检查命名规范
            naming_matches = 0
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
            for word in words:
                if re.match(rules.get('naming', r'^[a-zA-Z_][a-zA-Z0-9_]*$'), word):
                    naming_matches += 1
            quality_metrics['naming_score'] = naming_matches / len(words) if words else 0
            
            # 检查注释
            comment_lines = len(re.findall(rules.get('comments', r'//.*|/\*.*\*/'), code))
            quality_metrics['comment_score'] = min(comment_lines / len(lines), 1.0) if lines else 0
            
            # 检查结构
            structure_score = 0
            if language == "python":
                if 'def ' in code or 'class ' in code:
                    structure_score += 0.5
                if 'import ' in code or 'from ' in code:
                    structure_score += 0.3
                if '"""' in code or "'''" in code:
                    structure_score += 0.2
            elif language == "javascript":
                if 'function ' in code or '=>' in code:
                    structure_score += 0.5
                if 'import ' in code or 'const ' in code or 'let ' in code:
                    structure_score += 0.3
                if '//' in code or '/*' in code:
                    structure_score += 0.2
            elif language == "java":
                if 'public ' in code or 'private ' in code:
                    structure_score += 0.5
                if 'import ' in code:
                    structure_score += 0.3
                if '//' in code or '/*' in code:
                    structure_score += 0.2
            
            quality_metrics['structure_score'] = structure_score
            
            # 计算综合评分
            quality_metrics['overall_score'] = (
                quality_metrics['indentation_score'] * 0.2 +
                quality_metrics['naming_score'] * 0.3 +
                quality_metrics['comment_score'] * 0.2 +
                quality_metrics['structure_score'] * 0.3
            )
            
        except Exception as e:
            print(f"⚠️ 代码质量评估出错: {e}")
        
        return quality_metrics
    
    def generate_code(self, 
                     prompt: str, 
                     language: str = "python",
                     code_type: str = "function",
                     max_length: int = 500,
                     temperature: float = 0.7) -> Dict:
        """
        生成代码
        
        Args:
            prompt: 代码生成提示
            language: 编程语言 (python, javascript, java)
            code_type: 代码类型 (function, class, script, complete, comment)
            max_length: 最大代码长度
            temperature: 生成温度
            
        Returns:
            包含代码和元信息的字典
        """
        start_time = time.time()
        
        try:
            # 构建提示
            if language in self.code_prompts and code_type in self.code_prompts[language]:
                template = self.code_prompts[language][code_type]
            else:
                template = f"请用{language}编写代码，实现以下功能：\n\n"
            
            full_prompt = template + prompt
            
            # 编码输入
            input_ids = self.tokenizer.encode(full_prompt, max_length=self.max_length)
            input_tensor = Tensor([input_ids], mindspore.int32)
            
            # 分析专家使用情况
            expert_analysis = self._analyze_code_expert_usage(input_tensor, language)
            
            # 生成代码（简化版，实际需要完整的自回归生成）
            generated_code = self._simulate_code_generation(prompt, language, code_type, max_length, temperature)
            
            # 评估代码质量
            quality_metrics = self._evaluate_code_quality(generated_code, language)
            
            # 计算生成时间
            generation_time = time.time() - start_time
            
            return {
                'code': generated_code,
                'prompt': prompt,
                'language': language,
                'code_type': code_type,
                'expert_analysis': expert_analysis,
                'quality_metrics': quality_metrics,
                'generation_time': generation_time,
                'input_tokens': len(input_ids),
                'output_tokens': len(generated_code.split())
            }
            
        except Exception as e:
            print(f"❌ 代码生成失败: {e}")
            return {
                'code': f"# 代码生成失败: {str(e)}",
                'prompt': prompt,
                'language': language,
                'code_type': code_type,
                'expert_analysis': {},
                'quality_metrics': {},
                'generation_time': time.time() - start_time,
                'input_tokens': 0,
                'output_tokens': 0
            }
    
    def _simulate_code_generation(self, prompt: str, language: str, code_type: str, max_length: int, temperature: float) -> str:
        """模拟代码生成（用于演示）"""
        
        # 根据语言和类型生成示例代码
        if language == "python":
            if code_type == "function":
                return f'''def {prompt.lower().replace(" ", "_")}():
    """
    {prompt}
    """
    # TODO: 实现具体功能
    pass'''
            
            elif code_type == "class":
                return f'''class {prompt.replace(" ", "")}:
    """
    {prompt}
    """
    
    def __init__(self):
        # 初始化代码
        pass
    
    def process(self):
        # 处理逻辑
        pass'''
            
            elif code_type == "script":
                return f'''#!/usr/bin/env python3
"""
{prompt}
"""

import sys

def main():
    # 主函数逻辑
    print("Hello, World!")
    
if __name__ == "__main__":
    main()'''
            
            elif code_type == "complete":
                return f'''# 补全代码
{prompt}
    # 实现具体逻辑
    pass'''
            
            elif code_type == "comment":
                return f'''# {prompt}
# 这是一个示例代码，用于演示注释功能
# 可以根据实际需要添加更多注释'''
        
        elif language == "javascript":
            if code_type == "function":
                return f'''function {prompt.lower().replace(" ", "_")}() {{
    // {prompt}
    // TODO: 实现具体功能
    return null;
}}'''
            
            elif code_type == "class":
                return f'''class {prompt.replace(" ", "")} {{
    constructor() {{
        // 初始化代码
    }}
    
    process() {{
        // 处理逻辑
    }}
}}'''
            
            elif code_type == "complete":
                return f'''// 补全代码
{prompt}
    // 实现具体逻辑
    return null;'''
            
            elif code_type == "comment":
                return f'''// {prompt}
// 这是一个示例代码，用于演示注释功能
// 可以根据实际需要添加更多注释'''
        
        elif language == "java":
            if code_type == "function":
                return f'''public void {prompt.lower().replace(" ", "_")}() {{
    // {prompt}
    // TODO: 实现具体功能
}}'''
            
            elif code_type == "class":
                return f'''public class {prompt.replace(" ", "")} {{
    // {prompt}
    
    public {prompt.replace(" ", "")}() {{
        // 构造函数
    }}
    
    public void process() {{
        // 处理逻辑
    }}
}}'''
            
            elif code_type == "complete":
                return f'''// 补全代码
{prompt}
    // 实现具体逻辑
}}'''
            
            elif code_type == "comment":
                return f'''// {prompt}
// 这是一个示例代码，用于演示注释功能
// 可以根据实际需要添加更多注释'''
        
        # 默认返回
        return f"// {prompt}\n// 代码生成中..."
    
    def complete_code(self, partial_code: str, language: str = "python") -> Dict:
        """代码补全"""
        return self.generate_code(partial_code, language, "complete")
    
    def add_comments(self, code: str, language: str = "python") -> Dict:
        """添加代码注释"""
        return self.generate_code(code, language, "comment")
    
    def batch_generate(self, prompts: List[Tuple[str, str, str]]) -> List[Dict]:
        """批量生成代码"""
        results = []
        
        print(f"🔄 开始批量生成 {len(prompts)} 个代码...")
        
        for i, (prompt, language, code_type) in enumerate(prompts):
            print(f"  生成第 {i+1}/{len(prompts)} 个代码...")
            result = self.generate_code(prompt, language, code_type)
            results.append(result)
        
        print(f"✅ 批量生成完成")
        return results
    
    def visualize_code_analysis(self, results: List[Dict], save_path: str = None):
        """可视化代码分析结果"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 语言分布
            languages = [r['language'] for r in results]
            language_counts = {}
            for lang in languages:
                language_counts[lang] = language_counts.get(lang, 0) + 1
            
            ax1.pie(language_counts.values(), labels=language_counts.keys(), autopct='%1.1f%%')
            ax1.set_title('Programming Language Distribution')
            
            # 代码质量评分
            quality_scores = [r['quality_metrics'].get('overall_score', 0) for r in results]
            ax2.hist(quality_scores, bins=10, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Code Quality Score')
            ax2.set_ylabel('Number of Codes')
            ax2.set_title('Code Quality Distribution')
            ax2.axvline(np.mean(quality_scores), color='red', linestyle='--', label=f'Average: {np.mean(quality_scores):.2f}')
            ax2.legend()
            
            # 专家使用热力图
            expert_data = []
            for r in results:
                expert_dist = r['expert_analysis'].get('expert_distribution', [0.25] * 4)
                expert_data.append(expert_dist)
            
            if expert_data:
                expert_matrix = np.array(expert_data)
                im = ax3.imshow(expert_matrix.T, cmap='YlOrRd', aspect='auto')
                ax3.set_xlabel('Code Samples')
                ax3.set_ylabel('Expert ID')
                ax3.set_title('Expert Usage Heatmap')
                plt.colorbar(im, ax=ax3)
            
            # 性能指标
            generation_times = [r['generation_time'] for r in results]
            code_lengths = [r['quality_metrics'].get('code_length', 0) for r in results]
            
            ax4.scatter(code_lengths, generation_times, alpha=0.6)
            ax4.set_xlabel('Code Length')
            ax4.set_ylabel('Generation Time (s)')
            ax4.set_title('Code Length vs Generation Time')
            
            # 添加趋势线
            if len(code_lengths) > 1:
                z = np.polyfit(code_lengths, generation_times, 1)
                p = np.poly1d(z)
                ax4.plot(code_lengths, p(code_lengths), "r--", alpha=0.8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"📊 代码分析图已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ 可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_code_report(self, results: List[Dict], output_path: str = "code_generation_report.json"):
        """生成代码生成报告"""
        try:
            # 统计信息
            languages = [r['language'] for r in results]
            language_stats = {}
            for lang in set(languages):
                language_stats[lang] = languages.count(lang)
            
            quality_scores = [r['quality_metrics'].get('overall_score', 0) for r in results]
            generation_times = [r['generation_time'] for r in results]
            
            report = {
                'summary': {
                    'total_codes': len(results),
                    'language_distribution': language_stats,
                    'average_quality_score': np.mean(quality_scores),
                    'average_generation_time': np.mean(generation_times),
                    'total_input_tokens': sum([r['input_tokens'] for r in results]),
                    'total_output_tokens': sum([r['output_tokens'] for r in results])
                },
                'quality_analysis': {
                    'best_quality': max(quality_scores),
                    'worst_quality': min(quality_scores),
                    'quality_std': np.std(quality_scores),
                    'high_quality_codes': len([s for s in quality_scores if s > 0.8])
                },
                'performance_analysis': {
                    'fastest_generation': min(generation_times),
                    'slowest_generation': max(generation_times),
                    'generation_time_std': np.std(generation_times)
                },
                'results': results
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"📄 代码生成报告已保存: {output_path}")
            return report
            
        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            return None


def demo_code_generation_assistant():
    """演示代码生成助手"""
    print("="*80)
    print("💻 代码生成助手演示")
    print("="*80)
    
    # 初始化代码生成助手
    assistant = CodeGenerationAssistant()
    
    # 测试用例
    test_cases = [
        ("计算斐波那契数列", "python", "function"),
        ("实现快速排序算法", "python", "function"),
        ("创建一个学生管理类", "python", "class"),
        ("实现数组去重功能", "javascript", "function"),
        ("创建一个购物车组件", "javascript", "class"),
        ("实现字符串反转", "java", "function"),
        ("创建一个图书管理系统", "java", "class"),
    ]
    
    # 生成代码
    results = []
    for prompt, language, code_type in test_cases:
        print(f"\n💻 生成 {language} {code_type}: {prompt}")
        result = assistant.generate_code(prompt, language, code_type)
        results.append(result)
        
        # 打印结果
        print(f"   语言: {result['language']}")
        print(f"   类型: {result['code_type']}")
        print(f"   生成时间: {result['generation_time']:.3f} 秒")
        print(f"   质量评分: {result['quality_metrics'].get('overall_score', 0):.3f}")
        print(f"   代码长度: {result['quality_metrics'].get('code_length', 0)} 字符")
        print(f"   代码预览: {result['code'][:100]}...")
    
    # 代码补全示例
    print(f"\n🔧 代码补全示例:")
    partial_code = "def calculate_area(radius):\n    # 计算圆的面积\n    "
    completion_result = assistant.complete_code(partial_code, "python")
    print(f"   补全结果: {completion_result['code']}")
    
    # 添加注释示例
    print(f"\n📝 添加注释示例:")
    code_without_comments = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    comment_result = assistant.add_comments(code_without_comments, "python")
    print(f"   注释结果: {comment_result['code']}")
    
    # 可视化分析
    print(f"\n📊 生成代码分析图...")
    assistant.visualize_code_analysis(results, "code_analysis.png")
    
    # 生成报告
    print(f"\n📄 生成代码报告...")
    report = assistant.generate_code_report(results, "code_generation_report.json")
    
    # 打印统计信息
    if report:
        stats = report['summary']
        quality = report['quality_analysis']
        performance = report['performance_analysis']
        
        print(f"\n📈 统计信息:")
        print(f"   总代码数: {stats['total_codes']}")
        print(f"   语言分布: {stats['language_distribution']}")
        print(f"   平均质量评分: {stats['average_quality_score']:.3f}")
        print(f"   平均生成时间: {stats['average_generation_time']:.3f} 秒")
        print(f"   高质量代码数: {quality['high_quality_codes']}")
        print(f"   最快生成时间: {performance['fastest_generation']:.3f} 秒")
        print(f"   最慢生成时间: {performance['slowest_generation']:.3f} 秒")
    
    print(f"\n✅ 代码生成助手演示完成！")


if __name__ == "__main__":
    demo_code_generation_assistant()
