class SavedTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self._saved_version = tensor._version_counter.value

    def materialize(self):
        if self.tensor._version_counter.value != self._saved_version:
            raise RuntimeError(
                "one of the variables needed for gradient computation has been modified by an inplace operation"
            )
        return self.tensor


class Node:
    def __init__(self, backward, inputs):
        self.backward = backward
        self.inputs = inputs
        self._saved_tensors = []

    def save_for_backward(self, *tensors):
        self._saved_tensors = [SavedTensor(t) for t in tensors]

    def saved_tensors(self):
        return tuple(saved.materialize() for saved in self._saved_tensors)
