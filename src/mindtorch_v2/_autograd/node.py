from .graph import current_saved_tensors_hooks


class SavedTensor:
    def __init__(self, tensor):
        self._tensor_ref = tensor
        self._saved_version = tensor._version_counter.value
        self._released = False
        hooks = current_saved_tensors_hooks()
        self._hooks = hooks
        if hooks is None:
            self._packed = None
        else:
            pack, _ = hooks
            self._packed = pack(tensor)

    def release(self):
        self._released = True

    def materialize(self):
        if self._released:
            raise RuntimeError(
                "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). "
                "Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). "
                "Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."
            )
        if self._tensor_ref._version_counter.value != self._saved_version:
            shape = "x".join(str(d) for d in getattr(self._tensor_ref, "shape", ()))
            tensor_type = "torch.Tensor"
            op = "AsStridedBackward0"
            raise RuntimeError(
                "one of the variables needed for gradient computation has been modified by an inplace operation: "
                f"[{tensor_type} [{shape}]], which is output 0 of {op}, is at version {self._tensor_ref._version_counter.value}; "
                f"expected version {self._saved_version} instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, "
                "with torch.autograd.set_detect_anomaly(True)."
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

    def release_saved_tensors(self):
        for saved in self._saved_tensors:
            saved.release()
