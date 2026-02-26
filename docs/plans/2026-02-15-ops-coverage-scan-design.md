# MindTorch v2 Ops Coverage Scan Design

## Goal
Generate a repeatable, low-overhead report that lists PyTorch `aten::` ops not yet registered in MindTorch v2.

## Scope
- MindTorch v2 CPU/meta backend registrations under `src/mindtorch_v2/_backends/**`.
- PyTorch `aten::` ops from `torch._C._get_all_op_names()`.
- Output report for collaboration, not auto-implementation.

## Approach
1. **Static registry scan**
   - Walk all Python files under `src/mindtorch_v2/_backends/**`.
   - Parse each file with `ast` and extract literal string arguments to `registry.register(...)` and `registry.register_schema(...)`.
   - Normalize op names by ensuring an `aten::` prefix if missing.
   - Record any non-literal registry calls as unresolved for follow-up.

2. **Torch op collection**
   - Import `torch` and call `torch._C._get_all_op_names()`.
   - Filter to names beginning with `aten::`.

3. **Diff and report**
   - Compute `missing_ops = torch_aten_ops - mindtorch_v2_ops`.
   - Write a report file at `docs/plans/ops-coverage-aten-missing.md` with counts, missing list, and unresolved registry calls.

## Output
- Report includes:
  - total `aten::` ops in PyTorch
  - total registered ops in MindTorch v2 backends
  - missing ops list
  - unresolved registry calls

## Error Handling
- Syntax errors in scanned files or failure to import `torch` abort the scan with a clear message and non-zero exit.

## Workflow
- Run `python scripts/scan_ops_coverage.py` from repo root.
- Use the report to seed CPU op work and update `docs/plans/ops-coverage.md` as ops are implemented.
