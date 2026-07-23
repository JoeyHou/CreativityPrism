#!/usr/bin/env bash
# Adapter for TTCT (Torrance Test of Creative Thinking).
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

ALIAS="$(lookup_alias "$MODEL" ttct)"

TASK_DIR="$REPO_ROOT/tasks/ttct"
cd "$TASK_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Set HF_TOKEN from credentials file if not already in env.
if [[ -z "${HF_TOKEN:-}" && -f "${CREATIVITYPRISM_API_KEYS:-}" ]]; then
    HF_TOKEN="$(python3 -c "import json; print(json.load(open('$CREATIVITYPRISM_API_KEYS')).get('hf',''))" 2>/dev/null || true)"
    [[ -n "$HF_TOKEN" ]] && export HF_TOKEN
fi

# Default temperature for ttct (mirrors run_inference_apis_1.sh and friends).
TEMP="${CREATIVITYPRISM_TTCT_TEMP:-1}"

NUM_SAMPLES_ARG=()
[[ -n "$LIMIT" ]] && NUM_SAMPLES_ARG=(-num_samples "$LIMIT")

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    python3 ./src/inference/ttct_inference.py \
        -model_name "$ALIAS" \
        -temp "$TEMP" \
        -run_id "$RUN_ID" \
        "${NUM_SAMPLES_ARG[@]}"
fi

# Native output: data/outputs/<run_id>/<model_short>.json
MODEL_SHORT="${ALIAS##*/}"
NATIVE_OUT="$TASK_DIR/data/outputs/${RUN_ID}/${MODEL_SHORT}.json"
echo "OUTPUT_PATH=${NATIVE_OUT}"