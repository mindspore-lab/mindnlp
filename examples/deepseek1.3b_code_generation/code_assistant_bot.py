#!/usr/bin/env python
# coding=utf-8
"""
基于 DeepSeek Coder 模型的代码助手机器人
"""

import os
import argparse
import re
import time
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

class CodeAssistant:
    """代码助手类，使用 DeepSeek Coder 模型提供代码生成和解释服务"""

    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-base"):
        """初始化代码助手"""
        self.model_name = model_name
        
        # 加载模型和分词器
        console.print(f"正在加载 [bold]{model_name}[/bold] 模型...", style="yellow")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        console.print("模型加载完成!", style="green")
        
        # 对话历史
        self.conversation_history = []
        
        # 命令列表
        self.commands = {
            "/help": self.show_help,
            "/clear": self.clear_history,
            "/save": self.save_conversation,
            "/exit": lambda: "exit",
            "/examples": self.show_examples
        }

    def start(self):
        """启动交互式代码助手"""
        console.print(Panel.fit(
            "[bold]DeepSeek Coder 代码助手[/bold]\n\n"
            "一个基于 DeepSeek Coder 模型的代码生成和解释工具\n"
            "输入 [bold blue]/help[/bold blue] 查看帮助信息\n"
            "输入 [bold blue]/exit[/bold blue] 退出程序",
            title="欢迎使用",
            border_style="green"
        ))
        
        # 创建历史记录文件
        history_file = os.path.expanduser("~/.code_assistant_history")
        session = PromptSession(history=FileHistory(history_file),
                               auto_suggest=AutoSuggestFromHistory())
        
        while True:
            try:
                user_input = session.prompt("\n[用户] > ")
                
                # 处理命令
                if user_input.strip().startswith("/"):
                    command = user_input.strip().split()[0]
                    if command in self.commands:
                        result = self.commands[command]()
                        if result == "exit":
                            break
                        continue
                
                if not user_input.strip():
                    continue
                
                # 将用户输入添加到历史记录
                self.conversation_history.append(f"[用户] {user_input}")
                
                # 获取回复
                start_time = time.time()
                console.print("[AI 思考中...]", style="yellow")
                
                response = self.generate_response(user_input)
                
                # 提取代码块
                code_blocks = self.extract_code_blocks(response)
                
                # 格式化输出
                console.print("\n[AI 助手]", style="bold green")
                
                # 如果有代码块，特殊处理
                if code_blocks:
                    parts = re.split(r'```(?:\w+)?\n|```', response)
                    i = 0
                    for part in parts:
                        if part.strip():
                            if i % 2 == 0:  # 文本部分
                                console.print(Markdown(part.strip()))
                            else:  # 代码部分
                                lang = self.detect_language(code_blocks[(i-1)//2])
                                console.print(Syntax(code_blocks[(i-1)//2], lang, theme="monokai", 
                                                    line_numbers=True, word_wrap=True))
                        i += 1
                else:
                    # 没有代码块，直接显示为Markdown
                    console.print(Markdown(response))
                
                elapsed_time = time.time() - start_time
                console.print(f"[生成用时: {elapsed_time:.2f}秒]", style="dim")
                
                # 将回复添加到历史记录
                self.conversation_history.append(f"[AI] {response}")
                
            except KeyboardInterrupt:
                console.print("\n中断操作...", style="bold red")
                break
            except Exception as e:
                console.print(f"\n发生错误: {str(e)}", style="bold red")
    
    def generate_response(self, prompt, max_length=1000, temperature=0.7):
        """生成回复"""
        # 处理提示
        if "代码" in prompt or "函数" in prompt or "实现" in prompt or "编写" in prompt:
            # 检测是否已经包含了代码格式声明
            if not "```" in prompt:
                prompt = f"```python\n# {prompt}\n"
        
        inputs = self.tokenizer(prompt, return_tensors="ms")
        
        # 生成回复
        generated_ids = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
        )
        
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 清理响应，如果有的话
        if prompt in response:
            response = response.replace(prompt, "", 1).strip()
        
        return response
    
    def extract_code_blocks(self, text):
        """从文本中提取代码块"""
        pattern = r'```(?:\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches
    
    def detect_language(self, code):
        """简单检测代码语言"""
        if "def " in code and ":" in code:
            return "python"
        elif "{" in code and "}" in code and ";" in code:
            if "public class" in code or "private" in code:
                return "java"
            elif "function" in code or "var" in code or "let" in code or "const" in code:
                return "javascript"
            else:
                return "cpp"
        elif "<" in code and ">" in code and ("</" in code or "/>" in code):
            return "html"
        else:
            return "text"
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        # 可用命令:
        
        - `/help` - 显示此帮助信息
        - `/clear` - 清除当前对话历史
        - `/save` - 保存当前对话到文件
        - `/examples` - 显示示例提示
        - `/exit` - 退出程序
        
        # 使用技巧:
        
        1. 提供详细的需求描述以获得更好的代码生成效果
        2. 如果生成的代码不满意，可以要求修改或优化
        3. 可以请求解释已有代码或调试问题
        4. 对复杂功能，建议分步骤请求实现
        """
        console.print(Markdown(help_text))
    
    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []
        console.print("已清除对话历史", style="green")
    
    def save_conversation(self):
        """保存对话到文件"""
        if not self.conversation_history:
            console.print("没有对话内容可保存", style="yellow")
            return
        
        filename = f"code_assistant_conversation_{int(time.time())}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# DeepSeek Coder 代码助手对话记录\n\n")
            for entry in self.conversation_history:
                if entry.startswith("[用户]"):
                    f.write(f"## {entry}\n\n")
                else:
                    f.write(f"{entry[5:]}\n\n")
        
        console.print(f"对话已保存到 {filename}", style="green")
    
    def show_examples(self):
        """显示示例提示"""
        examples = """
        # 示例提示:
        
        1. "实现一个Python函数，计算两个日期之间的工作日数量"
        
        2. "编写一个简单的Flask API，具有用户注册和登录功能"
        
        3. "创建一个二分查找算法的JavaScript实现"
        
        4. "使用pandas分析CSV数据并生成统计报告"
        
        5. "实现一个简单的React组件，显示待办事项列表"
        
        6. "解释以下代码的功能: 
           ```python
           def mystery(arr):
               return [x for x in arr if x == x[::-1]]
           ```"
        
        7. "优化下面的排序算法:
           ```python
           def sort(arr):
               for i in range(len(arr)):
                   for j in range(len(arr)):
                       if arr[i] < arr[j]:
                           arr[i], arr[j] = arr[j], arr[i]
               return arr
           ```"
        """
        console.print(Markdown(examples))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepSeek Coder 代码助手")
    parser.add_argument("--model", type=str, default="deepseek-ai/deepseek-coder-1.3b-base",
                        help="使用的模型名称或路径")
    args = parser.parse_args()
    
    # 创建并启动代码助手
    assistant = CodeAssistant(model_name=args.model)
    assistant.start()


if __name__ == "__main__":
    main() 