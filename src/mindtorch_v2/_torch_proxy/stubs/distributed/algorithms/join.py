"""Stub for torch.distributed.algorithms.join module."""


class Join:
    """Join context manager for joining distributed processes."""

    def __init__(self, joinables, enable=True, throw_on_early_termination=False,
                 divide_by_initial_world_size=True):
        self.joinables = joinables
        self.enable = enable
        self.throw_on_early_termination = throw_on_early_termination
        self.divide_by_initial_world_size = divide_by_initial_world_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @staticmethod
    def notify_join_context(device):
        """Notify join context of a device."""
        pass


class Joinable:
    """Base class for joinable processes."""

    def __init__(self):
        pass

    def join_device(self):
        return None

    def join_process_group(self):
        return None

    def join(self):
        pass


class JoinHook:
    """Hook for join operations."""

    def __init__(self, joinable):
        self.joinable = joinable

    def main_hook(self):
        pass

    def post_hook(self, is_last_joiner):
        pass


__all__ = ['Join', 'Joinable', 'JoinHook']
