import os
import numpy as np
import acl
from mindtorch_v2._backends import ascend_ctypes


def main():
    ret = acl.init()
    print("acl.init", ret)
    ret = acl.rt.set_device(0)
    print("set_device", ret)
    ctx, ret = acl.rt.create_context(0)
    print("create_context", ret)
    stream, ret = acl.rt.create_stream()
    print("create_stream", ret)

    bindings = ascend_ctypes.get_bindings()
    print("bindings ok")

    a = np.array([-1.0, 2.0], dtype=np.float32)
    size = a.nbytes
    a_dev, ret = acl.rt.malloc(size, 0)
    print("malloc", ret)
    ret = acl.rt.memcpy(a_dev, size, acl.util.numpy_to_ptr(a), size, 1)
    print("memcpy", ret)
    out_dev, ret = acl.rt.malloc(size, 0)
    print("malloc out", ret)

    try:
        ascend_ctypes.relu(a_dev, out_dev, a.shape, (1,), "float32", stream)
        print("relu ok")
    except Exception as e:
        print("relu error", e)

    acl.rt.free(a_dev)
    acl.rt.free(out_dev)
    acl.rt.destroy_stream(stream)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
