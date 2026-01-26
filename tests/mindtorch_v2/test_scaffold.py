def test_import():
    import mindtorch_v2 as torch
    assert hasattr(torch, '__version__')
