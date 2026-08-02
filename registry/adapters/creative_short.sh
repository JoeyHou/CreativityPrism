#!/usr/bin/env bash
# Adapter for the Creative Short Story task.
source "$(dirname "$0")/_common.sh"
parse_adapter_args "$@"
activate_env modern

ALIAS="$(lookup_alias "$MODEL" aut_ttcw_cshort)"
API_ID="$(lookup_alias "$MODEL" api_call 2>/dev/null || echo "$ALIAS")"

TASK_DIR="$REPO_ROOT/tasks/aut_ttcw_cshort"
TMP_CONFIG="$(mktemp -t cshort_cfg_XXXXXX.json)"
TMP_EVAL_CONFIG="$(mktemp -t cshort_eval_cfg_XXXXXX.json)"
trap 'rm -f "$TMP_CONFIG" "$TMP_EVAL_CONFIG"' EXIT

TEST_SIZE_LINE=""
[[ -n "$LIMIT" ]] && TEST_SIZE_LINE='"test_size": '"$LIMIT"','

cat > "$TMP_CONFIG" <<JSON
{
    "experiments_list": [
        {
            ${TEST_SIZE_LINE}
            "run_id": "${RUN_ID}/creative_short/${ALIAS}",
            "task": "creative_short",
            "model_name": "${API_ID}"
        }
    ]
}
JSON

cd "$TASK_DIR"
NATIVE_OUT="$TASK_DIR/data/output/${RUN_ID}/creative_short/${ALIAS}"
if [[ "$MODE" == "inference" || "$MODE" == "both" ]]; then
    python run_inference.py "$TMP_CONFIG"
    emit_artifact inference "$NATIVE_OUT"
fi

if [[ "$MODE" == "eval" || "$MODE" == "both" ]]; then
    # creative_short evaluation is fully automated (DSI / n-gram diversity / novelty /
    # theme uniqueness) and uses no LLM judge. model_name is carried for config-shape
    # parity with the other bundled tasks; --judge-model has no effect here.
    JUDGE_ID="$(lookup_alias "$JUDGE" api_call 2>/dev/null || lookup_alias "$JUDGE" aut_ttcw_cshort)"
    cat > "$TMP_EVAL_CONFIG" <<JSON
{
    "experiments_list": [
        {
            "run_id": "${RUN_ID}/creative_short/${ALIAS}",
            "task": "creative_short",
            "model_name": "${JUDGE_ID}"
        }
    ]
}
JSON
    python run_evaluation.py "$TMP_EVAL_CONFIG"
    emit_artifact eval "$NATIVE_OUT/eval_output_cleaned.json"
fi