import sys
import types

import mindtorch_v2._backends.npu.acl_loader as acl_loader


def test_ensure_acl_caches_module(monkeypatch):
    calls = {"import": 0}

    def fake_import_acl():
        calls["import"] += 1
        return types.SimpleNamespace(name="acl")

    monkeypatch.setattr(acl_loader, "_import_acl", fake_import_acl)
    monkeypatch.setattr(acl_loader, "_ACL_READY", False)
    monkeypatch.setattr(acl_loader, "_ACL_MODULE", None)

    first = acl_loader.ensure_acl()
    second = acl_loader.ensure_acl()

    assert first is second
    assert calls["import"] == 1


def test_ensure_acl_retries_after_failure(monkeypatch):
    calls = {"import": 0}

    def fake_import_acl():
        calls["import"] += 1
        if calls["import"] == 1:
            raise RuntimeError("boom")
        return types.SimpleNamespace(name="acl")

    monkeypatch.setattr(acl_loader, "_import_acl", fake_import_acl)
    monkeypatch.setattr(acl_loader, "_ACL_READY", False)
    monkeypatch.setattr(acl_loader, "_ACL_MODULE", None)

    try:
        acl_loader.ensure_acl()
    except RuntimeError:
        pass

    acl = acl_loader.ensure_acl()
    assert acl.name == "acl"
    assert calls["import"] == 2



def test_ensure_acl_appends_python_path_on_import_error(monkeypatch):
    calls = {"import": 0}
    added = []

    def fake_append_python_path(paths):
        added.extend(paths)

    def fake_import_acl():
        calls["import"] += 1
        if calls["import"] == 1:
            raise ModuleNotFoundError("No module named 'acl'")
        return types.SimpleNamespace(name="acl")

    monkeypatch.setattr(acl_loader, "_import_acl", fake_import_acl)
    monkeypatch.setattr(acl_loader, "_append_python_path", fake_append_python_path)
    monkeypatch.setattr(acl_loader, "_ACL_READY", False)
    monkeypatch.setattr(acl_loader, "_ACL_MODULE", None)

    acl = acl_loader.ensure_acl()
    assert acl.name == "acl"
    assert calls["import"] == 2
    assert added
