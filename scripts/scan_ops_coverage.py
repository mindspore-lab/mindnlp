#!/usr/bin/env python
"""Scan mindtorch v2 registered ops and diff against PyTorch aten ops."""
from __future__ import annotations

import ast
import pathlib
import sys
from typing import Dict, Iterable, List, Set, Tuple


def _iter_py_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*.py"):
        if path.name.startswith("__init__"):
            # Still parse __init__ modules, they often register ops.
            yield path
        else:
            yield path


def _normalize_op_name(name: str) -> str:
    if "::" in name:
        return name
    return f"aten::{name}"


def _base_op_name(name: str) -> str:
    if "." in name:
        return name.split(".", 1)[0]
    return name


def _extract_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_registry_ops(py_file: pathlib.Path) -> Tuple[Set[str], List[str]]:
    text = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(py_file))
    except SyntaxError as exc:
        raise SyntaxError(f"{py_file}: {exc}") from exc

    ops: Set[str] = set()
    unresolved: List[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in {"register", "register_schema"}:
            continue
        if not node.args:
            continue
        op_name = _extract_string(node.args[0])
        if op_name is None:
            unresolved.append(f"{py_file}:{getattr(node, 'lineno', '?')}")
            continue
        ops.add(_normalize_op_name(op_name))
    return ops, unresolved


def _collect_mindtorch_ops(root: pathlib.Path) -> Tuple[Set[str], List[str]]:
    ops: Set[str] = set()
    unresolved: List[str] = []
    for path in _iter_py_files(root):
        file_ops, file_unresolved = _collect_registry_ops(path)
        ops.update(file_ops)
        unresolved.extend(file_unresolved)
    return ops, unresolved


def _collect_torch_aten_ops() -> Set[str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch: {exc}") from exc
    if hasattr(torch._C, "_get_all_op_names"):
        names = torch._C._get_all_op_names()
    elif hasattr(torch._C, "_dispatch_get_all_op_names"):
        names = torch._C._dispatch_get_all_op_names()
    else:
        raise RuntimeError("No supported torch._C op listing API found")
    return {_base_op_name(name) for name in names if name.startswith("aten::")}


def _is_tensor_type(type_str: str) -> bool:
    if type_str.startswith("TensorOptions"):
        return False
    return type_str.startswith("Tensor")


def _schema_tags(schema) -> Set[str]:
    args = schema.arguments
    rets = schema.returns
    arg_types = [str(arg.type) for arg in args]
    ret_types = [str(ret.type) for ret in rets]

    has_tensor_inputs = any(_is_tensor_type(t) for t in arg_types)
    returns_tensor = any(_is_tensor_type(t) for t in ret_types)
    has_generator = any("Generator" in t for t in arg_types)
    has_dim = any(arg.name in {"dim", "dims"} for arg in args)
    has_keepdim = any(arg.name == "keepdim" for arg in args)
    has_shape = any("int[]" in t or "SymInt[]" in t for t in arg_types)
    has_out = any(getattr(arg, "is_out", False) for arg in args)
    has_write = any(getattr(arg, "is_write", False) for arg in args)

    tags: Set[str] = set()
    if has_out or has_write:
        tags.add("inplace_or_out")
    if not has_tensor_inputs and returns_tensor:
        tags.add("creation")
    if has_generator:
        tags.add("random")
    if has_dim or has_keepdim or (has_tensor_inputs and not returns_tensor):
        tags.add("reduction")
    if has_shape:
        tags.add("view_or_shape")
    if (
        has_tensor_inputs
        and returns_tensor
        and "reduction" not in tags
        and "view_or_shape" not in tags
        and "creation" not in tags
    ):
        tags.add("elementwise")
    return tags


def _classify_missing_ops(missing_ops: Iterable[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    try:
        from torch._C import _jit_get_schemas_for_operator
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch schema API: {exc}") from exc

    categories: Dict[str, List[str]] = {
        "random": [],
        "creation": [],
        "reduction": [],
        "view_or_shape": [],
        "elementwise": [],
        "other": [],
    }
    inplace_or_out: List[str] = []

    for op in missing_ops:
        try:
            schemas = _jit_get_schemas_for_operator(op)
        except Exception:
            categories["other"].append(op)
            continue

        tags: Set[str] = set()
        for schema in schemas:
            tags.update(_schema_tags(schema))

        if "inplace_or_out" in tags:
            inplace_or_out.append(op)

        if "random" in tags:
            categories["random"].append(op)
        elif "creation" in tags:
            categories["creation"].append(op)
        elif "reduction" in tags:
            categories["reduction"].append(op)
        elif "view_or_shape" in tags:
            categories["view_or_shape"].append(op)
        elif "elementwise" in tags:
            categories["elementwise"].append(op)
        else:
            categories["other"].append(op)

    return categories, sorted(set(inplace_or_out))


def _write_raw_report(
    report_path: pathlib.Path,
    torch_ops: Set[str],
    mindtorch_ops: Set[str],
    missing_ops: List[str],
    unresolved: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# MindTorch v2 vs PyTorch aten ops\n")
    lines.append(f"Torch aten ops: {len(torch_ops)}\n")
    lines.append(f"MindTorch v2 registered ops: {len(mindtorch_ops)}\n")
    lines.append(f"Missing ops: {len(missing_ops)}\n")
    lines.append("\n## Missing aten ops\n")
    for name in missing_ops:
        lines.append(f"- `{name}`")
    if unresolved:
        lines.append("\n## Unresolved registry calls\n")
        lines.append("These registry calls used non-literal op names and were skipped.")
        for entry in unresolved:
            lines.append(f"- `{entry}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_grouped_report(
    report_path: pathlib.Path,
    torch_ops: Set[str],
    mindtorch_ops: Set[str],
    missing_ops: List[str],
    categories: Dict[str, List[str]],
    inplace_or_out: List[str],
    unresolved: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# MindTorch v2 vs PyTorch aten ops (Grouped)\n")
    lines.append(f"Torch aten ops: {len(torch_ops)}\n")
    lines.append(f"MindTorch v2 registered ops: {len(mindtorch_ops)}\n")
    lines.append(f"Missing ops: {len(missing_ops)}\n")
    lines.append("\n## Grouped missing ops (primary category)\n")
    for name, ops in categories.items():
        lines.append(f"### {name.replace('_', ' ').title()} ({len(ops)})")
        for op in sorted(ops):
            lines.append(f"- `{op}`")
        lines.append("")
    lines.append("## Missing ops with in-place/out variants\n")
    for op in inplace_or_out:
        lines.append(f"- `{op}`")
    if unresolved:
        lines.append("\n## Unresolved registry calls\n")
        lines.append("These registry calls used non-literal op names and were skipped.")
        for entry in unresolved:
            lines.append(f"- `{entry}`")
    lines.append("\n## Notes\n")
    lines.append("Schema grouping is best-effort and driven only by operator schemas.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    backend_root = repo_root / "src" / "mindtorch_v2" / "_backends"
    if not backend_root.exists():
        print(f"Backend root not found: {backend_root}", file=sys.stderr)
        return 1

    try:
        mindtorch_ops, unresolved = _collect_mindtorch_ops(backend_root)
    except SyntaxError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        torch_ops = _collect_torch_aten_ops()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    missing_ops = sorted(torch_ops - mindtorch_ops)
    grouped_report_path = repo_root / "docs" / "plans" / "ops-coverage-aten-missing-grouped.md"
    raw_report_path = repo_root / "docs" / "plans" / "ops-coverage-aten-missing.md"
    categories, inplace_or_out = _classify_missing_ops(missing_ops)
    _write_raw_report(raw_report_path, torch_ops, mindtorch_ops, missing_ops, unresolved)
    _write_grouped_report(
        grouped_report_path,
        torch_ops,
        mindtorch_ops,
        missing_ops,
        categories,
        inplace_or_out,
        unresolved,
    )

    print("Ops coverage report generated:")
    print(f"  {raw_report_path}")
    print(f"  {grouped_report_path}")
    print("Summary:")
    print(f"  Torch aten ops: {len(torch_ops)}")
    print(f"  MindTorch v2 registered ops: {len(mindtorch_ops)}")
    print(f"  Missing ops: {len(missing_ops)}")
    if unresolved:
        print(f"  Unresolved registry calls: {len(unresolved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
