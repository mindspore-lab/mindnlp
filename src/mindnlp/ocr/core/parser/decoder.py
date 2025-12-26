"""
Token解码器
将模型输出的Token IDs解码为文本
"""

from transformers import PreTrainedTokenizer
from utils.logger import get_logger


logger = get_logger(__name__)


class TokenDecoder:
    """Token解码器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        初始化Token解码器
        
        Args:
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer
        logger.info("TokenDecoder initialized")
    
    def decode(self, token_ids) -> str:
        """
        解码Token IDs
        
        Args:
            token_ids: Token IDs数组
            
        Returns:
            str: 解码后的文本
        """
        # 使用tokenizer解码
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        logger.debug(f"Decoded text length: {len(text)}")
        return text
