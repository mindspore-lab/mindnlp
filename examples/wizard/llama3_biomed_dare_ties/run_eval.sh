#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Evaluate Llama3-merge-biomed-8b on Ascend NPU
# Benchmarks match the Open LLM Leaderboard v1 settings.
# Each dataset produces a separate log + JSON result file.
#
# Usage:
#   bash run_eval.sh                                   # 默认模型路径
#   bash run_eval.sh /path/to/merged_model             # 自定义模型路径
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINDNLP_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_DIR="${1:-${SCRIPT_DIR}/output/merged}"
EVAL_DTYPE="bfloat16"
BATCH_SIZE="${BATCH_SIZE:-1}"

EVAL_ROOT="${SCRIPT_DIR}/output/eval"
mkdir -p "$EVAL_ROOT"

# ---- Environment ----
export PYTHONUNBUFFERED=1
export PYTHONPATH="${MINDNLP_ROOT}/src:${PYTHONPATH:-}"

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
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[warn] HF_TOKEN not set and .hf_token empty — gated datasets may fail."
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Dataset cache directory
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRIPT_DIR}/output/datasets}"
mkdir -p "$HF_DATASETS_CACHE"

LM_EVAL_SCRIPT="${SCRIPT_DIR}/run_lm_eval.py"
MODEL_ARGS="pretrained=${MODEL_DIR},dtype=${EVAL_DTYPE}"

# ---- Tasks: ordered fastest → slowest ----
# Format: task_name:num_fewshot
TASKS=(
  "mmlu_medical_genetics:5"
  "mmlu_anatomy:5"
  "mmlu_college_biology:5"
  "mmlu_college_medicine:5"
  "mmlu_clinical_knowledge:5"
  "mmlu_professional_medicine:5"
  "gsm8k:5"
  "arc_challenge:25"
  "winogrande:5"
  "hellaswag:10"
)

SUMMARY_LOG="${SCRIPT_DIR}/output/eval_summary.log"
echo "==== Eval start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" | tee "$SUMMARY_LOG"
echo "model_dir   = ${MODEL_DIR}" | tee -a "$SUMMARY_LOG"
echo "model_args  = ${MODEL_ARGS}" | tee -a "$SUMMARY_LOG"
echo "batch_size  = ${BATCH_SIZE}" | tee -a "$SUMMARY_LOG"
echo "" | tee -a "$SUMMARY_LOG"

for entry in "${TASKS[@]}"; do
  IFS=':' read -r task nshot <<< "$entry"
  out_log="${EVAL_ROOT}/${task}.log"

  echo "---- [${task}] num_fewshot=${nshot} start $(date -u +%H:%M:%S) ----" | tee -a "$SUMMARY_LOG"

  set +e
  python "$LM_EVAL_SCRIPT" \
    --model mindspore \
    --model_args "${MODEL_ARGS}" \
    --tasks "${task}" \
    --num_fewshot "${nshot}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${EVAL_ROOT}/${task}.json" 2>&1 | tee "$out_log"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [ $exit_code -eq 0 ]; then
    result_line=$(grep -E "acc|exact_match" "$out_log" | tail -1 || echo "parse failed")
    echo "[OK]    task=${task} exit=${exit_code} ${result_line}" | tee -a "$SUMMARY_LOG"
  else
    echo "[FAIL]  task=${task} exit=${exit_code}" | tee -a "$SUMMARY_LOG"
  fi
  echo "" | tee -a "$SUMMARY_LOG"
done

echo "==== Eval done $(date -u +%Y-%m-%dT%H:%M:%SZ) ====" | tee -a "$SUMMARY_LOG"
