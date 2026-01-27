"""Stub for torch.distributed.algorithms module."""

from . import join

__all__ = ['join', 'Join']


class Join:
    """Join context manager stub."""
    def __init__(self, joinables, enable=True, throw_on_early_termination=False,
                 divide_by_initial_world_size=True):
        self.joinables = joinables
        self.enable = enable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def notify_join_context(device):
        pass
