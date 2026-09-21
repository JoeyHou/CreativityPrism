#!/usr/bin/env bash
# Overnight batch: two models across all eight tasks, inference then evaluation.
#
# Each stage is a separate process so a crash in one cannot take the others with
# it, and inside a stage the entry points already isolate per experiment. Every
# stage's exit code is recorded; the script keeps going and reports at the end.
#
# Run from tasks/prism_v1:  bash run_batch.sh 2>&1 | tee /tmp/batch.log
set -u

# Everything below has a default that works on a fresh clone and can be
# overridden from the environment. It used to hardcode one machine's paths,
# which made this script the one file a new user could not run.
REPO="${CREATIVITYPRISM_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY="${CREATIVITYPRISM_PYTHON:-python}"

export CREATIVITYPRISM_API_KEYS="${CREATIVITYPRISM_API_KEYS:-$REPO/api_keys.json}"
# Only the dat evaluation needs this; see tasks/prism_v1/README for how to obtain it.
export CREATIVITYPRISM_GLOVE_PATH="${CREATIVITYPRISM_GLOVE_PATH:-$REPO/data/glove.840B.300d.txt}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HF_HOME/.triton}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/.torch}"
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
# One card. Every vLLM path derives tensor_parallel_size from what is visible,
# and an unmasked SSH session sees all four including other people's.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

RESULTS=()
stage() {
    local name="$1"; shift
    echo ""
    echo "############ $name  ($(date '+%H:%M:%S')) ############"
    "$@"
    local rc=$?
    RESULTS+=("$name rc=$rc")
    echo "############ $name done rc=$rc ($(date '+%H:%M:%S')) ############"
}

# Pre-flight. A model that cannot be reached should stop the batch here, not
# four hours in.
echo "=== pre-flight ==="
# model_hub lives at the repo root, not in this bundle.
(cd "$REPO" && $PY -m model_hub.smoke --models GPT4.1-mini --judges GPT4.1-mini) || {
    echo "pre-flight failed; not starting the batch"; exit 1; }

stage "inference/api"   $PY run_inference.py  configs/batch/inference_api.json
stage "inference/open"  $PY run_inference.py  configs/batch/inference_open.json
stage "eval/api"        $PY run_evaluation.py configs/batch/evaluation_gpt41mini.json
stage "eval/open"       $PY run_evaluation.py configs/batch/evaluation_qwen7b.json

echo ""
echo "=================== BATCH SUMMARY ==================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "Outputs under tasks/prism_v1/data/output/b1/"
