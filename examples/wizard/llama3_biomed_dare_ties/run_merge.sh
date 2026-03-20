#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DARE-TIES merge — reproduces lighteternal/Llama3-merge-biomed-8b
# Uses MindNLP Wizard merge backend.
#
# Usage:
#   bash run_merge.sh                          # 使用默认输出路径
#   bash run_merge.sh /path/to/output_dir      # 自定义输出路径
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECIPE="${SCRIPT_DIR}/dare_ties_biomed.yaml"
OUT_DIR="${1:-${SCRIPT_DIR}/output/merged}"
LOG="${SCRIPT_DIR}/output/merge.log"

# Auto-detect mindnlp src root (examples/wizard/llama3_biomed_dare_ties/ -> src/)
MINDNLP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${MINDNLP_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "$(dirname "$LOG")"

# Ascend NPU runtime
# Override ASCEND_HOME if CANN is not installed at the default /usr/local/Ascend
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend}"
if [ -f "${ASCEND_HOME}/ascend-toolkit/set_env.sh" ]; then
    source "${ASCEND_HOME}/ascend-toolkit/set_env.sh"
fi
if [ -d "${ASCEND_HOME}/driver/lib64" ]; then
    export LD_LIBRARY_PATH="${ASCEND_HOME}/driver/lib64:${ASCEND_HOME}/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
fi

# Activate Wizard conda environment
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate Wizard 2>/dev/null || echo "[warn] conda env 'Wizard' not found, using current env"
fi

# Load HF token from .hf_token file if HF_TOKEN not set
HF_TOKEN_FILE="${SCRIPT_DIR}/../.hf_token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    _token=$(grep -v '^#' "$HF_TOKEN_FILE" | grep -v '^$' | head -1 | tr -d '[:space:]')
    if [ -n "$_token" ]; then
        export HF_TOKEN="$_token"
        export HUGGING_FACE_HUB_TOKEN="$_token"
        export HF_HUB_TOKEN="$_token"
    fi
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PYTHONUNBUFFERED=1

echo "==== DARE-TIES merge start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" | tee "$LOG"
echo "Recipe : ${RECIPE}" | tee -a "$LOG"
echo "Output : ${OUT_DIR}" | tee -a "$LOG"
echo "PYTHONPATH: ${PYTHONPATH}" | tee -a "$LOG"

python -m mindnlp.wizard.merge.scripts.run_yaml \
    "${RECIPE}" \
    "${OUT_DIR}" \
    --copy-tokenizer \
    --write-model-card 2>&1 | tee -a "$LOG"

echo "==== DARE-TIES merge done $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" | tee -a "$LOG"
echo "Merged model saved to: ${OUT_DIR}"
