"""
Prompt构建器
根据任务类型、输出格式和语言构建合适的Prompt
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict
from utils.logger import get_logger
from config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()


class PromptBuilder:
    """
    Prompt构建器

    功能:
    1. 从 YAML 文件加载 Prompt 模板
    2. 支持多种任务类型 (general/document/table/formula)
    3. 支持多种输出格式 (text/json/markdown)
    4. 支持多语言 (zh/en/ja/ko/auto)
    5. 支持自定义 Prompt
    6. 模板变量替换
    """

    def __init__(self, template_file: Optional[str] = None):
        """
        初始化Prompt构建器

        Args:
            template_file: Prompt 模板文件路径（默认使用 config/prompts.yaml）
        """
        if template_file is None:
            # 默认使用项目根目录下的 prompts.yaml
            template_file = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"

        self.template_file = template_file
        self.prompt_templates = self._load_prompt_templates()
        logger.info(f"PromptBuilder initialized with template file: {template_file}")

    def build(
        self,
        task_type: str = "general",
        output_format: str = "text",
        language: str = "auto",
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        构建Prompt

        Args:
            task_type: 任务类型 (general/document/table/formula)
            output_format: 输出格式 (text/json/markdown)
            language: 语言设置 (auto/zh/en/ja/ko)
            custom_prompt: 自定义Prompt (如果提供则优先使用)
            **kwargs: 额外的模板变量

        Returns:
            str: 构建的Prompt
        """
        # 如果提供了自定义 Prompt，应用模板变量并返回
        if custom_prompt:
            logger.info("Using custom prompt")
            return self._replace_variables(custom_prompt, **kwargs)

        # 获取任务类型对应的 Prompt 模板
        task_prompt = self._get_task_prompt(task_type, language)

        # 添加输出格式说明
        format_prompt = self._get_format_prompt(output_format, language)

        # 添加语言提示
        language_prompt = self._get_language_prompt(language)

        # 组合完整 Prompt
        full_prompt = self._combine_prompts(task_prompt, format_prompt, language_prompt)

        # 模板变量替换
        full_prompt = self._replace_variables(full_prompt, **kwargs)

        logger.debug(f"Built prompt (task={task_type}, format={output_format}, lang={language}): {full_prompt[:100]}...")
        return full_prompt

    def _load_prompt_templates(self) -> Dict:
        """
        从 YAML 文件加载 Prompt 模板

        Returns:
            dict: Prompt 模板字典
        """
        try:
            # 尝试从 YAML 文件加载
            if os.path.exists(self.template_file):
                with open(self.template_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                logger.info(f"Loaded prompt templates from {self.template_file}")

                # 检查 YAML 结构：如果有 task_prompts 键，说明是分离结构
                if 'task_prompts' in yaml_data or 'format_prompts' in yaml_data:
                    # 转换为期望的格式
                    templates = {}
                    task_prompts = yaml_data.get('task_prompts', {})
                    format_prompts = yaml_data.get('format_prompts', {})
                    language_prompts = yaml_data.get('language_prompts', {})

                    # 合并任务提示
                    for task_type, lang_dict in task_prompts.items():
                        templates[task_type] = lang_dict

                    # 添加格式提示（特殊键）
                    if format_prompts:
                        templates['__format__'] = format_prompts

                    # 添加语言提示（特殊键）
                    if language_prompts:
                        templates['__language__'] = language_prompts

                    return templates
                else:
                    # 已经是期望格式
                    return yaml_data
            else:
                logger.warning(f"Template file not found: {self.template_file}, using default templates")
                return self._get_default_templates()

        except Exception as e:
            logger.error(f"Failed to load prompt templates: {e}, using default templates")
            return self._get_default_templates()

    def _get_default_templates(self) -> Dict:
        """
        获取默认 Prompt 模板（作为后备）

        Returns:
            dict: 默认模板字典
        """
        return {
            "general": {
                "zh": "请识别图像中的所有文本内容",
                "en": "Please recognize all text content in the image",
                "ja": "画像内のすべてのテキストコンテンツを認識してください",
                "ko": "이미지의 모든 텍스트 내용을 인식하십시오",
                "auto": "请识别图像中的所有文本内容"
            },
            "document": {
                "zh": "请解析文档的结构和内容，保持原有格式",
                "en": "Please parse the document structure and content, maintaining the original format",
                "ja": "ドキュメントの構造と内容を解析し、元の形式を維持してください",
                "ko": "문서의 구조와 내용을 분석하고 원래 형식을 유지하십시오",
                "auto": "请解析文档的结构和内容，保持原有格式"
            },
            "table": {
                "zh": "请提取图像中的表格数据，保持表格结构",
                "en": "Please extract table data from the image, maintaining the table structure",
                "ja": "画像からテーブルデータを抽出し、テーブル構造を維持してください",
                "ko": "이미지에서 테이블 데이터를 추출하고 테이블 구조를 유지하십시오",
                "auto": "请提取图像中的表格数据，保持表格结构"
            },
            "formula": {
                "zh": "请识别图像中的数学公式，以LaTeX格式输出",
                "en": "Please recognize mathematical formulas in the image, output in LaTeX format",
                "ja": "画像内の数式を認識し、LaTeX形式で出力してください",
                "ko": "이미지의 수학 공식을 인식하고 LaTeX 형식으로 출력하십시오",
                "auto": "请识别图像中的数学公式，以LaTeX格式输出"
            },
            "output_format": {
                "text": {
                    "zh": "，以纯文本格式输出",
                    "en": ", output in plain text format",
                    "ja": "、プレーンテキスト形式で出力してください",
                    "ko": ", 일반 텍스트 형식으로 출력하십시오",
                    "auto": "，以纯文本格式输出"
                },
                "json": {
                    "zh": "，以JSON格式输出，包含文本内容和位置坐标",
                    "en": ", output in JSON format with text content and position coordinates",
                    "ja": "、テキスト内容と位置座標を含むJSON形式で出力してください",
                    "ko": ", 텍스트 내용과 위치 좌표를 포함하는 JSON 형식으로 출력하십시오",
                    "auto": "，以JSON格式输出，包含文本内容和位置坐标"
                },
                "markdown": {
                    "zh": "，以Markdown格式输出，保持文档结构",
                    "en": ", output in Markdown format, maintaining document structure",
                    "ja": "、ドキュメント構造を維持しながらMarkdown形式で出力してください",
                    "ko": ", 문서 구조를 유지하면서 Markdown 형식으로 출력하십시오",
                    "auto": "，以Markdown格式输出，保持文档结构"
                }
            },
            "language": {
                "zh": "。图像中主要包含中文文本",
                "en": ". The image mainly contains English text",
                "ja": "。画像には主に日本語のテキストが含まれています",
                "ko": ". 이미지에는 주로 한국어 텍스트가 포함되어 있습니다",
                "auto": ""
            }
        }

    def _get_task_prompt(self, task_type: str, language: str = "auto") -> str:
        """
        获取任务类型对应的 Prompt

        Args:
            task_type: 任务类型
            language: 语言

        Returns:
            str: 任务 Prompt
        """
        if task_type not in self.prompt_templates:
            logger.warning(f"Unknown task type: {task_type}, using 'general'")
            task_type = "general"

        task_prompts = self.prompt_templates[task_type]
        return task_prompts.get(language, task_prompts.get("auto", ""))

    def _get_format_prompt(self, output_format: str, language: str = "auto") -> str:
        """
        获取输出格式 Prompt

        Args:
            output_format: 输出格式
            language: 语言

        Returns:
            str: 格式 Prompt
        """
        # 尝试从特殊键 __format__ 获取
        format_section = self.prompt_templates.get("__format__",
                                                   self.prompt_templates.get("output_format", {}))
        format_prompts = format_section.get(output_format, {})

        if isinstance(format_prompts, dict):
            return format_prompts.get(language, format_prompts.get("auto", ""))
        return ""

    def _get_language_prompt(self, language: str) -> str:
        """
        获取语言提示 Prompt

        Args:
            language: 语言

        Returns:
            str: 语言 Prompt
        """
        if language == "auto":
            return ""

        # 尝试从特殊键 __language__ 获取
        language_section = self.prompt_templates.get("__language__",
                                                     self.prompt_templates.get("language", {}))
        return language_section.get(language, "")

    def _combine_prompts(
        self,
        task_prompt: str,
        format_prompt: str,
        language_prompt: str
    ) -> str:
        """
        组合所有 Prompt 片段

        Args:
            task_prompt: 任务 Prompt
            format_prompt: 格式 Prompt
            language_prompt: 语言 Prompt

        Returns:
            str: 完整 Prompt
        """
        return f"{task_prompt}{format_prompt}{language_prompt}。"

    def _replace_variables(self, prompt: str, **kwargs) -> str:
        """
        替换 Prompt 中的模板变量

        Args:
            prompt: Prompt 字符串
            **kwargs: 变量字典

        Returns:
            str: 替换后的 Prompt
        """
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Variable {e} not found in kwargs, leaving placeholder")
            return prompt
