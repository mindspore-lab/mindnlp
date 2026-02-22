import mindtorch_v2._functional as F


def test_functional_exports_present():
    missing = [
        name
        for name in (
            "stack",
            "cat",
            "concat",
            "hstack",
            "vstack",
            "column_stack",
            "pad_sequence",
            "block_diag",
            "cartesian_prod",
            "chunk",
            "split",
            "unbind",
        )
        if not hasattr(F, name)
    ]
    assert not missing, f"Missing _functional exports: {missing}"
