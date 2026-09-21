#!/usr/bin/env bash
# Adapter for NeoCoder (denial prompting).
#
# All eight tasks now live in tasks/prism_v1 behind the same two entrypoints, so every
# adapter has the same shape: translate the runner's flags into one ephemeral
# config and hand it to run_inference.py / run_evaluation.py. What differs
# between tasks is only the JSON below.
#
# --judge-model is a NO-OP for the default stage: correctness is decided by
# executing the code against test cases. Set CREATIVITYPRISM_NEOCODER_STAGE to
# techniques or creativity for the judged stages.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

# One alias key for the whole repo: "all". Before the port each bundle had its
# own key and two of them were truncated (mistral_7b_instr, llama3_8b_instr),
# which resolved to nothing.
ALIAS="$(lookup_alias "$MODEL" all)"

TASK_DIR="$REPO_ROOT/tasks/prism_v1"
TMP_CONFIG="$(mktemp -t neocoder_cfg_XXXXXX.json)"
TMP_EVAL_CONFIG="$(mktemp -t neocoder_eval_cfg_XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG" "$TMP_EVAL_CONFIG"' EXIT

TEST_SIZE_LINE=""
[[ -n "$LIMIT" ]] && TEST_SIZE_LINE='"test_size": '"$LIMIT"','

RUN_PATH="${RUN_ID}/neocoder/${ALIAS}"
NATIVE_OUT="$TASK_DIR/data/output/${RUN_PATH}"
cd "$TASK_DIR"

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    cat > "$TMP_CONFIG" <<JSON
{
    "experiments_list": [
        {
            ${TEST_SIZE_LINE}
            "run_id": "${RUN_PATH}",
            "task": "neocoder",
            "model_name": "${ALIAS}",
            "input_data_dir": "${DATA_DIR}/neocoder/NeoCoder.json",
            "dp_rounds": 5,
            "temperature": 0.75,
            "top_p": 1,
            "max_tokens": 4096
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
            "task": "neocoder",
            "eval_input_dir": "data/output/${RUN_PATH}/inference_output.json",
            "test_case_path": "${DATA_DIR}/neocoder/test_cases_annotated.json",
            "code_extractor": "fenced",
            "stage": "${CREATIVITYPRISM_NEOCODER_STAGE:-correctness}"
        }
    ]
}
JSON
    run_task run_evaluation.py "$TMP_EVAL_CONFIG"
    emit_artifact eval "$NATIVE_OUT/eval_output_cleaned.json"
fi
