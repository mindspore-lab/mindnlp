#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-mindtorch311}"

env PYTHONPATH=src conda run -n "${ENV_NAME}" python -m pytest -q tests/mindtorch_v2/test_310b_watchlist_npu.py
