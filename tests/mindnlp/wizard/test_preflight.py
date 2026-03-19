import pytest

from mindnlp.wizard.merge.config import MergeConfiguration
from mindnlp.wizard.merge.config import InputModelDefinition
from mindnlp.wizard.merge.options import MergeOptions
from mindnlp.wizard.merge.preflight import run_merge_preflight


def test_preflight_runs_for_gta_family_half_dtype():
    cfg = MergeConfiguration(
        merge_method="della",
        dtype="bfloat16",
        base_model="dummy/base",
        models=[
            InputModelDefinition(model="dummy/a", parameters={"weight": 1.0}),
            InputModelDefinition(model="dummy/b", parameters={"weight": 1.0}),
        ],
    )
    run_merge_preflight(cfg, MergeOptions(device="CPU"))


def test_preflight_is_skipped_for_non_half_dtype():
    cfg = MergeConfiguration(
        merge_method="della",
        dtype="float32",
        base_model="dummy/base",
        models=[
            InputModelDefinition(model="dummy/a", parameters={"weight": 1.0}),
            InputModelDefinition(model="dummy/b", parameters={"weight": 1.0}),
        ],
    )
    run_merge_preflight(cfg, MergeOptions(device="CPU"))


def test_preflight_surfaces_probe_failure(monkeypatch):
    import mindnlp.wizard.merge.preflight as preflight_mod

    cfg = MergeConfiguration(
        merge_method="della",
        dtype="bfloat16",
        base_model="dummy/base",
        models=[
            InputModelDefinition(model="dummy/a", parameters={"weight": 1.0}),
            InputModelDefinition(model="dummy/b", parameters={"weight": 1.0}),
        ],
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("probe boom")

    monkeypatch.setattr(preflight_mod, "_probe_half_precision_math", _boom)
    with pytest.raises(RuntimeError, match="probe boom"):
        run_merge_preflight(cfg, MergeOptions(device="CPU"))

