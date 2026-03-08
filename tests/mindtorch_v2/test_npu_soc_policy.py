from mindtorch_v2._backends.npu import ops_soc


def test_soc_policy_profile_mapping_contains_expected_profiles():
    assert ops_soc.fallback_ops("910a") == frozenset()
    assert ops_soc.fallback_ops("910b") == frozenset()
    assert ops_soc.fallback_ops("310p") == frozenset()
    assert "uniform_" in ops_soc.fallback_ops("310b")


def test_soc_policy_case_insensitive_profile_name():
    assert ops_soc.use_fallback("uniform_", profile="310B")
    assert not ops_soc.use_fallback("uniform_", profile="910A")


def test_soc_policy_unknown_profile_returns_safe_default():
    assert ops_soc.fallback_ops("unknown") == frozenset()
    assert not ops_soc.use_fallback("uniform_", profile="unknown")


def test_soc_capability_table_routes_smallop_arange_for_310b_only():
    assert ops_soc.use_smallop_arange_1d(profile="310b")
    assert not ops_soc.use_smallop_arange_1d(profile="910a")
    assert not ops_soc.use_smallop_arange_1d(profile="910b")
    assert not ops_soc.use_smallop_arange_1d(profile="310p")


def test_soc_capability_unknown_profile_uses_default_value():
    assert not ops_soc.capability("use_smallop_arange_1d", profile="unknown")
    assert ops_soc.capability("missing_key", profile="unknown", default=True)


def test_soc_capability_table_routes_smallop_linspace_for_310b_only():
    assert ops_soc.use_smallop_linspace(profile="310b")
    assert not ops_soc.use_smallop_linspace(profile="910a")
    assert not ops_soc.use_smallop_linspace(profile="910b")
    assert not ops_soc.use_smallop_linspace(profile="310p")
