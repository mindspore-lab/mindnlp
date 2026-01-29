"""Dummy _dynamo module for compatibility."""


class _DisabledModule:
    """A class that can be used in isinstance checks but always returns False."""
    pass


class OptimizedModule(_DisabledModule):
    """Stub for torch._dynamo.eval_frame.OptimizedModule."""
    pass
