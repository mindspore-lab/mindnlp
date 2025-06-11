#!/usr/bin/env python
# coding=utf-8
"""
使用 DeepSeek Coder 模型生成代码示例
"""

import argparse
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="使用 DeepSeek Coder 生成代码")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-base",
        help="预训练模型名称或路径，默认为 deepseek-coder-1.3b-base",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="编写一个Python函数，实现快速排序算法",
        help="用于生成代码的提示文本",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=500,
        help="生成的最大长度",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成的温度，较高的值会使输出更加随机，较低的值使其更加集中和确定",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="nucleus采样的概率阈值",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="取前k个候选的限制",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 加载模型和分词器
    print(f"加载模型和分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    # 准备提示
    prompt = args.prompt
    
    # 添加前缀以获得更好的代码生成效果
    if not prompt.startswith("```"):
        if "python" in prompt.lower():
            prompt = f"```python\n# {prompt}\n"
        else:
            prompt = f"```python\n# {prompt}\n"
    
    # 分词
    inputs = tokenizer(prompt, return_tensors="ms")
    
    # 生成代码
    print(f"使用提示：'{args.prompt}' 生成代码...")
    generated_ids = model.generate(
        inputs.input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=True,
    )
    
    # 解码生成的代码
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # 打印生成的代码
    print("\n生成的代码:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    # 提取代码部分（如果有```标记的话）
    if "```" in generated_text:
        code_start = generated_text.find("```") + 3
        language_end = generated_text.find("\n", code_start)
        code_end = generated_text.find("```", language_end)
        if code_end == -1:  # 如果没有结束的```
            code = generated_text[language_end+1:]
        else:
            code = generated_text[language_end+1:code_end].strip()
        
        print("\n提取的纯代码:")
        print("-" * 50)
        print(code)
        print("-" * 50)

if __name__ == "__main__":
    main() 