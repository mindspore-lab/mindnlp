from .graph import current_saved_tensors_hooks


class SavedTensor:
    def __init__(self, tensor):
        self._tensor_ref = tensor
        self._saved_version = tensor._version_counter.value
        hooks = current_saved_tensors_hooks()
        self._hooks = hooks
        if hooks is None:
            self._packed = None
        else:
            pack, _ = hooks
            self._packed = pack(tensor)

    def materialize(self):
        if self._tensor_ref._version_counter.value != self._saved_version:
            raise RuntimeError(
                "one of the variables needed for gradient computation has been modified by an inplace operation"
            )
        if self._hooks is None:
            return self._tensor_ref
        _, unpack = self._hooks
        return unpack(self._packed)


class Node:
    def __init__(self, backward, inputs):
        self.backward = backward
        self.inputs = inputs
        self._saved_tensors = []

    def save_for_backward(self, *tensors):
        self._saved_tensors = [SavedTensor(t) for t in tensors]

    def saved_tensors(self):
        return tuple(saved.materialize() for saved in self._saved_tensors)
