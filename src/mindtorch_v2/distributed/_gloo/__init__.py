"""Gloo backend for CPU distributed training."""

from ._process_group_gloo import ProcessGroupGloo

__all__ = ["ProcessGroupGloo"]
