import tempfile

import mindtorch_v2 as torch
import torch as pt
from tests.mindtorch_v2.contract.helpers import assert_torch_error


def test_storage_resize_file_backed_error():
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(b"\x00" * 8)
        tmp.flush()
        x = torch.UntypedStorage.from_file(tmp.name, shared=False)
        px = pt.UntypedStorage.from_file(tmp.name, shared=False)

        def mt():
            x.resize_(0)

        def th():
            px.resize_(0)

        assert_torch_error(mt, th)
