"""
Prompt构建器
根据任务类型、输出格式和语言构建合适的Prompt
"""

from typing import Optional
from utils.logger import get_logger
from config.settings import get_settings


logger = get_logger(__name__)
settings = get_settings()


class PromptBuilder:
    """Prompt构建器"""
    
    def __init__(self):
        """初始化Prompt构建器"""
        self.prompt_templates = self._load_prompt_templates()
        logger.info("PromptBuilder initialized")
    
    def build_prompt(
        self,
        task_type: str = "general",
        output_format: str = "text",
        language: str = "auto",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        构建Prompt
        
        Args:
            task_type: 任务类型 (general/document/table/formula)
            output_format: 输出格式 (text/json/markdown)
            language: 语言设置 (auto/zh/en/ja/ko)
            custom_prompt: 自定义Prompt (如果提供则优先使用)
            
        Returns:
            str: 构建的Prompt
        """
        if custom_prompt:
            logger.info("Using custom prompt")
            return custom_prompt
        
        # 获取任务类型对应的Prompt模板
        task_prompt = self._get_task_prompt(task_type)
        
        # 添加输出格式说明
        format_prompt = self._get_format_prompt(output_format)
        
        # 添加语言提示
        language_prompt = self._get_language_prompt(language)
        
        # 组合完整Prompt
        full_prompt = self._combine_prompts(task_prompt, format_prompt, language_prompt)
        
        logger.debug(f"Built prompt: {full_prompt}")
        return full_prompt
    
    def _load_prompt_templates(self) -> dict:
        """
        加载Prompt模板
        实际应从config/prompts.yaml加载
        """
        return {
            "general": {
                "zh": "请识别图像中的所有文本内容",
                "en": "Please recognize all text content in the image",
                "auto": "请识别图像中的所有文本内容"
            },
            "document": {
                "zh": "请解析文档的结构和内容，保持原有格式",
                "en": "Please parse the document structure and content, maintaining the original format",
                "auto": "请解析文档的结构和内容，保持原有格式"
            },
            "table": {
                "zh": "请提取图像中的表格数据",
                "en": "Please extract table data from the image",
                "auto": "请提取图像中的表格数据"
            },
            "formula": {
                "zh": "请识别图像中的数学公式",
                "en": "Please recognize mathematical formulas in the image",
                "auto": "请识别图像中的数学公式"
            }
        }
    
    def _get_task_prompt(self, task_type: str) -> str:
        """获取任务类型Prompt"""
        if task_type not in self.prompt_templates:
            logger.warning(f"Unknown task type: {task_type}, using 'general'")
            task_type = "general"
        
        return self.prompt_templates[task_type]["auto"]
    
    def _get_format_prompt(self, output_format: str) -> str:
        """获取输出格式Prompt"""
        format_prompts = {
            "text": "，以纯文本格式输出",
            "json": "，以JSON格式输出，包含文本内容和位置坐标",
            "markdown": "，以Markdown格式输出，保持文档结构"
        }
        return format_prompts.get(output_format, "")
    
    def _get_language_prompt(self, language: str) -> str:
        """获取语言提示Prompt"""
        if language == "auto":
            return ""
        
        language_prompts = {
            "zh": "。图像中主要包含中文文本",
            "en": ". The image mainly contains English text",
            "ja": "。画像には主に日本語のテキストが含まれています",
            "ko": ". 이미지에는 주로 한국어 텍스트가 포함되어 있습니다"
        }
        return language_prompts.get(language, "")
    
    def _combine_prompts(self, task_prompt: str, format_prompt: str, language_prompt: str) -> str:
        """组合所有Prompt片段"""
        return f"{task_prompt}{format_prompt}{language_prompt}。"
