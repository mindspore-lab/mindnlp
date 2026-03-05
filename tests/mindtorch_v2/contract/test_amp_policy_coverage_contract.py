import mindtorch_v2.amp.policy as policy


def test_policy_lists_cover_all_categories():
    lists = policy.policy_lists()
    assert "torch_fp16" in lists
    assert "torch_fp32" in lists
    assert "torch_need_autocast_promote" in lists
    assert "torch_expect_builtin_promote" in lists
    assert "nn_fp16" in lists
    assert "nn_fp32" in lists
    assert "linalg_fp16" in lists
    assert "methods_fp16" in lists
    assert "methods_fp32" in lists
    assert "banned" in lists


def test_policy_map_includes_all_ops():
    lists = policy.policy_lists()
    all_ops = set()
    for ops in lists.values():
        all_ops.update(ops)
    # internal mapping should include every op in lists
    missing = [op for op in all_ops if policy._POLICY_MAP.get(op) is None]
    assert missing == []
