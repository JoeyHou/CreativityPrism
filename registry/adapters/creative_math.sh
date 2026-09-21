#!/usr/bin/env bash
# Adapter for Creative Math.
#
# All eight tasks now live in tasks/prism_v1 behind the same two entrypoints, so every
# adapter has the same shape: translate the runner's flags into one ephemeral
# config and hand it to run_inference.py / run_evaluation.py. What differs
# between tasks is only the JSON below.
#
# --judge-model is a NO-OP: correctness is decided by a single named judge
# (Claude, per the paper) and novelty by the judge panel below, not by the
# runner's --judge-model.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

# One alias key for the whole repo: "all". Before the port each bundle had its
# own key and two of them were truncated (mistral_7b_instr, llama3_8b_instr),
# which resolved to nothing.
ALIAS="$(lookup_alias "$MODEL" all)"

TASK_DIR="$REPO_ROOT/tasks/prism_v1"
TMP_CONFIG="$(mktemp -t creative_math_cfg_XXXXXX.json)"
TMP_EVAL_CONFIG="$(mktemp -t creative_math_eval_cfg_XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG" "$TMP_EVAL_CONFIG"' EXIT

TEST_SIZE_LINE=""
[[ -n "$LIMIT" ]] && TEST_SIZE_LINE='"test_size": '"$LIMIT"','

RUN_PATH="${RUN_ID}/creative_math/${ALIAS}"
NATIVE_OUT="$TASK_DIR/data/output/${RUN_PATH}"
cd "$TASK_DIR"

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    cat > "$TMP_CONFIG" <<JSON
{
    "experiments_list": [
        {
            ${TEST_SIZE_LINE}
            "run_id": "${RUN_PATH}",
            "task": "creative_math",
            "model_name": "${ALIAS}",
            "input_data_dir": "${DATA_DIR}/creative_math/creative_math.json",
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 2000
        }
    ]
}
JSON
    run_task run_inference.py "$TMP_CONFIG"
    emit_artifact inference "$NATIVE_OUT/inference_output.json"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    cat > "$TMP_EVAL_CONFIG" <<JSON
{
    "experiments_list": [
        {
            "run_id": "${RUN_PATH}",
            "task": "creative_math",
            "eval_input_dir": "data/output/${RUN_PATH}/inference_output.json",
            "judges": ["claude-sonnet-4-5-20250929", "gpt-4.1"],
            "judge_credentials": {"claude-sonnet-4-5-20250929": "claude-3-7-sonnet-20250219"},
            "correctness_rule": "single",
            "correctness_judge": "claude-sonnet-4-5-20250929",
            "fine_grained": false,
            "judge_temperature": 0.0,
            "judge_max_tokens": 1024
        }
    ]
}
JSON
    run_task run_evaluation.py "$TMP_EVAL_CONFIG"
    emit_artifact eval "$NATIVE_OUT/eval_output_cleaned.json"
fi
