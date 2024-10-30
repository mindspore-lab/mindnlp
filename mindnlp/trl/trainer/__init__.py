"""trainer init file."""
# from .base import BaseTrainer
from .dpo_trainer import DPOTrainer, _build_tokenized_answer, _truncate_tokens
from .dpo_config import DPOConfig, FDivergenceType
