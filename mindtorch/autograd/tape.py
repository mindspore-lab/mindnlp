import logging
from mindspore.common.api import _pynative_executor
from mindspore.ops import GradOperation

import mindtorch

grad_ = GradOperation(False, True, True)

def tape_func(): pass

class GradientTape(object):
    """Record operations for automatic differentiation.

    Operations are recorded if they are executed within this context manager and
    at least one of their inputs is being "watched".

    Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
    where `trainable=True` is default in both cases) are automatically watched.
    Tensors can be manually watched by invoking the `watch` method on this context
    manager.

    Note that only tensors with real or complex dtypes are differentiable.
    """

    def __init__(self, persistent=False, watch_accessed_variables=True):
        """Creates a new GradientTape.

        Args:
          persistent: Boolean controlling whether a persistent gradient tape
            is created. False by default, which means at most one call can
            be made to the gradient() method on this object.
          watch_accessed_variables: Boolean controlling whether the tape will
            automatically `watch` any (trainable) variables accessed while the tape
            is active. Defaults to True meaning gradients can be requested from any
            result computed in the tape derived from reading a trainable `Variable`.
            If False users must explicitly `watch` any `Variable`s they want to
            request gradients from.
        """
        self._tape = None
        self._persistent = persistent
        self._watch_accessed_variables = watch_accessed_variables
        self._watched_variables = ()
        self._recording = False

    def __enter__(self):
        """Enters a context inside which operations are recorded on this tape."""
        self._push_tape()
        return self

    def __exit__(self, typ, value, traceback):
        """Exits the recording context, no further operations are traced."""
        if self._recording:
            self._pop_tape()

    def _push_tape(self):
        """Pushes a new tape onto the tape stack."""
        if self._recording:
            raise ValueError(
                "Tape is still recording, This can happen if you try to "
                "re-enter an already-active tape."
            )
        _pynative_executor.set_grad_flag(True)
        _pynative_executor.new_graph(tape_func)
        self._recording = True

    def _pop_tape(self):
        if not self._recording:
            raise ValueError("Tape is not recording.")
        self._recording = False

    def _ensure_recording(self):
        """Ensures that this tape is recording."""
        if not self._recording:
            try:
                self._push_tape()
                yield
            finally:
                self._pop_tape()
        else:
            yield

    def stop_recording(self):
        """Temporarily stops recording operations on this tape.

        Operations executed while this context manager is active will not be
        recorded on the tape. This is useful for reducing the memory used by tracing
        all computations.

        For example:

        >>> x = tf.constant(4.0)
        >>> with tf.GradientTape() as tape:
        ...   with tape.stop_recording():
        ...     y = x ** 2
        >>> dy_dx = tape.gradient(y, x)
        >>> print(dy_dx)
        None

        Yields:
          None
        Raises:
          RuntimeError: if the tape is not currently recording.
        """
        if self._tape is None:
            raise RuntimeError(
                "Trying to stop recording a tape which is not recording."
            )
        self._pop_tape()
        try:
            yield
        finally:
            self._push_tape()

    def reset(self):
        """Clears all information stored in this tape.

        Equivalent to exiting and reentering the tape context manager with a new
        tape. For example, the two following code blocks are equivalent:

        ```
        with tf.GradientTape() as t:
          loss = loss_fn()
        with tf.GradientTape() as t:
          loss += other_loss_fn()
        t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn


        # The following is equivalent to the above
        with tf.GradientTape() as t:
          loss = loss_fn()
          t.reset()
          loss += other_loss_fn()
        t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
        ```

        This is useful if you don't want to exit the context manager for the tape,
        or can't because the desired reset point is inside a control flow construct:

        ```
        with tf.GradientTape() as t:
          loss = ...
          if loss > k:
            t.reset()
        ```
        """
        self._pop_tape()
        self._tape = None
        self._push_tape()

    def watched_variables(self):
        """Returns variables watched by this tape in order of construction."""
        if self._tape is not None:
            self._watched_variables = self._tape.watched_variables()
        return self._watched_variables

    def gradient(
        self,
        target,
        sources,
        output_gradients=None,
    ):
        """Computes the gradient using operations recorded in context of this tape.

        Note: Unless you set `persistent=True` a GradientTape can only be used to
        compute one set of gradients (or jacobians).

        In addition to Tensors, gradient also supports RaggedTensors. For example,

        >>> x = tf.ragged.constant([[1.0, 2.0], [3.0]])
        >>> with tf.GradientTape() as g:
        ...   g.watch(x)
        ...   y = x * x
        >>> g.gradient(y, x)
        <tf.RaggedTensor [[2.0, 4.0], [6.0]]>

        Args:
          target: a list or nested structure of Tensors or Variables or
            CompositeTensors to be differentiated.
          sources: a list or nested structure of Tensors or Variables or
            CompositeTensors. `target` will be differentiated against elements in
            `sources`.
          output_gradients: a list of gradients, one for each differentiable
            element of target. Defaults to None.
          unconnected_gradients: a value which can either hold 'none' or 'zero' and
            alters the value which will be returned if the target and sources are
            unconnected. The possible values and effects are detailed in
            'UnconnectedGradients' and it defaults to 'none'.

        Returns:
          a list or nested structure of Tensors (or IndexedSlices, or None, or
          CompositeTensor), one for each element in `sources`. Returned structure
          is the same as the structure of `sources`.

        Raises:
          RuntimeError: If called on a used, non-persistent tape.
          RuntimeError: If called inside the context of the tape.
          TypeError: If the target is a None object.
          ValueError: If the target is a variable or if unconnected gradients is
           called with an unknown value.
        """
        if target.shape == ():
            gradient = mindtorch.tensor(1, dtype=target.dtype, device=target.device)
        else:
            raise RuntimeError("grad must specified for non-0-tensor")

        _pynative_executor.end_graph(tape_func, target.data)
        weights = list(sources)
        _pynative_executor.check_run(grad_, tape_func, weights, None, gradient)
        grads = _pynative_executor.grad(tape_func, grad_, weights, None, gradient)
        return grads
