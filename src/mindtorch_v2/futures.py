"""Minimal Future implementation for DDP communication hooks."""


class Future:
    """A simple Future class for async operations.

    This is a synchronous implementation that immediately resolves.
    Sufficient for DDP comm hooks since our allreduce is synchronous.
    """

    def __init__(self):
        self._result = None
        self._done = False
        self._callbacks = []

    def set_result(self, result):
        """Set the result and mark as done."""
        self._result = result
        self._done = True
        # Execute callbacks
        for callback in self._callbacks:
            self._result = callback(self)

    def wait(self):
        """Wait for the future to complete and return result."""
        return self._result

    def value(self):
        """Return result as a list (PyTorch compatibility)."""
        return [self._result]

    def then(self, callback):
        """Chain a callback to execute after completion.

        Args:
            callback: Function that takes a Future and returns a new result

        Returns:
            A new Future with the callback result
        """
        new_future = Future()
        if self._done:
            # Already done, execute immediately
            result = callback(self)
            new_future.set_result(result)
        else:
            # Store callback to execute when done
            self._callbacks.append(lambda fut: callback(fut))
        return new_future

    def done(self):
        """Check if the future is done."""
        return self._done
