"""
API服务层模块
提供RESTful API接口
"""

from .app import create_app, get_engine

__all__ = ['create_app', 'get_engine']
