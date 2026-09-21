#!/usr/bin/env bash
# Adapter for the Creativity Index (DJ Search).
#
# All eight tasks now live in tasks/prism_v1 behind the same two entrypoints, so every
# adapter has the same shape: translate the runner's flags into one ephemeral
# config and hand it to run_inference.py / run_evaluation.py. What differs
# between tasks is only the JSON below.
#
# --judge-model is a NO-OP: scoring is an n-gram lookup against infini-gram, not
# an LLM judge. One sweep is hundreds of HTTP round trips; the run checkpoints
# after every sweep, so re-running the same run_id resumes.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

# One alias key for the whole repo: "all". Before the port each bundle had its
# own key and two of them were truncated (mistral_7b_instr, llama3_8b_instr),
# which resolved to nothing.
ALIAS="$(lookup_alias "$MODEL" all)"

TASK_DIR="$REPO_ROOT/tasks/prism_v1"
TMP_CONFIG="$(mktemp -t creativity_index_cfg_XXXXXX.json)"
TMP_EVAL_CONFIG="$(mktemp -t creativity_index_eval_cfg_XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG" "$TMP_EVAL_CONFIG"' EXIT

TEST_SIZE_LINE=""
[[ -n "$LIMIT" ]] && TEST_SIZE_LINE='"test_size": '"$LIMIT"','

RUN_PATH="${RUN_ID}/creativity_index/${ALIAS}"
NATIVE_OUT="$TASK_DIR/data/output/${RUN_PATH}"
cd "$TASK_DIR"

if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    cat > "$TMP_CONFIG" <<JSON
{
    "experiments_list": [
        {
            ${TEST_SIZE_LINE}
            "run_id": "${RUN_PATH}",
            "task": "creativity_index",
            "model_name": "${ALIAS}",
            "input_data_dir": "${DATA_DIR}/creativity_index",
            "domains": ["book", "poem", "speech"],
            "temperature": 1,
            "top_p": 0.9,
            "max_tokens": 288
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
            "task": "creativity_index",
            "eval_input_dir": "data/output/${RUN_PATH}/inference_output.json",
            "corpus": "v4_dolma-v1_7_llama",
            "lm_tokenizer": true,
            "num_workers": 2,
            "request_delay": 0.15
        }
    ]
}
JSON
    run_task run_evaluation.py "$TMP_EVAL_CONFIG"
    emit_artifact eval "$NATIVE_OUT/eval_output_cleaned.json"
fi
