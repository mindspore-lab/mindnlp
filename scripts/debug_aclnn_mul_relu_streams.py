import numpy as np
import acl
from mindtorch_v2._backends import ascend_ctypes


def main():
    acl.init()
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)
    stream1, _ = acl.rt.create_stream()
    stream2, _ = acl.rt.create_stream()

    a = np.array([-1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    size = a.nbytes
    a_dev, _ = acl.rt.malloc(size, 0)
    b_dev, _ = acl.rt.malloc(size, 0)
    out_dev, _ = acl.rt.malloc(size, 0)
    relu_out, _ = acl.rt.malloc(size, 0)
    acl.rt.memcpy(a_dev, size, acl.util.numpy_to_ptr(a), size, 1)
    acl.rt.memcpy(b_dev, size, acl.util.numpy_to_ptr(b), size, 1)

    ascend_ctypes.mul(a_dev, b_dev, out_dev, a.shape, (1,), "float32", stream1)
    print("mul done")
    ascend_ctypes.relu(a_dev, relu_out, a.shape, (1,), "float32", stream2)
    print("relu done")

    acl.rt.free(a_dev)
    acl.rt.free(b_dev)
    acl.rt.free(out_dev)
    acl.rt.free(relu_out)
    acl.rt.destroy_stream(stream1)
    acl.rt.destroy_stream(stream2)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
