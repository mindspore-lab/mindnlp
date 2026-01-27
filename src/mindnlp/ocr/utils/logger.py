"""
日志工具
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取logger实例

    Args:
        name: logger名称
        level: 日志级别

    Returns:
        logging.Logger: logger实例
    """
    logger = logging.getLogger(name)

    # 如果logger已经有处理器，直接返回
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(console_handler)

    return logger
